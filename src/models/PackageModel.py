"""
    ShelfGuard-VLM PackageModel

    Two-stage (cascade) shoplifting detection capsule:
      Stage 1 (edge)  : tracked pose detections + zone mapping + hand-product-concealment
                        interaction encoding + temporal classifier  ->  candidate events
      Stage 2 (VLM)   : short anonymised clip of each candidate is verified by a
                        Vision-Language Model which returns a decision and a reason.

    Expected flow on NovaVision:
      VideoFeed -> Yolo (Pose) -> ObjectTracking (ByteTrack) -> ShelfGuard -> VideoView / Alarm
      (optional) Yolo (Detection: product, bag, basket) -> ShelfGuard.inputObjects

    The model is written bottom-up: data models -> inputs/outputs -> config params
    -> executor configs -> request/response -> executors -> ConfigExecutor -> PackageModel.
"""

from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config,
    Model, Image, Detection, KeyPoints, BoundingBox, ROI,
)


# =====================================================================================
# Data models
# =====================================================================================

class KeyPoints(KeyPoints):
    confidence: Optional[float] = None


class TrackedPerson(Detection):
    """Pose detection coming from Yolo(Pose) -> ObjectTracking."""
    trackerID: Optional[Union[List, int]] = None
    keyPoints: Optional[List[KeyPoints]] = None
    imgUID: Optional[str] = None
    UUID: Optional[str] = ""
    source: Optional[str] = ""


class ShelfGuardPerson(TrackedPerson):
    """Tracked person enriched with stage-1 interaction state."""
    zones: Optional[List[str]] = None
    handState: Optional[str] = None
    suspicionScore: Optional[float] = None
    eventState: Optional[str] = None


class InteractionStep(Model):
    """One step of the 'hand in shelf -> hand with product -> hand in concealment' sequence."""
    phase: str
    hand: str
    startTime: float
    endTime: float


class CandidateEvent(Model):
    """Stage-1 output: a suspicious event candidate waiting for VLM verification."""
    eventID: str
    trackerID: int
    startTime: float
    endTime: float
    score: float
    interactionSequence: List[InteractionStep] = []
    structuralSummary: str = ""
    boundingBox: Optional[BoundingBox] = None
    clipFrameKeys: List[str] = []
    imgUID: Optional[str] = None


class TheftAlarm(Model):
    """Stage-2 output: VLM-verified event with a natural-language justification."""
    eventID: str
    trackerID: int
    decision: str
    confidence: float
    evidenceStart: float
    evidenceEnd: float
    reason: str
    language: str
    notify: bool
    stage1Score: Optional[float] = None
    latencyMs: Optional[float] = None
    boundingBox: Optional[BoundingBox] = None
    clipFrameKeys: List[str] = []
    imgUID: Optional[str] = None


# =====================================================================================
# Inputs
# =====================================================================================

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class InputDetections(Input):
    """Tracked pose detections (keyPoints + trackerID). Connect the ObjectTracking output here."""
    name: Literal["inputDetections"] = "inputDetections"
    value: List[TrackedPerson]
    type: Literal["list"] = "list"

    class Config:
        title = "Tracked Poses"


class InputObjects(Input):
    """Optional object detections (product, bag, basket ...) used for hand-product interaction."""
    name: Literal["inputObjects"] = "inputObjects"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Objects"


class InputCandidates(Input):
    """Candidate events produced by the CandidateDetection executor."""
    name: Literal["inputCandidates"] = "inputCandidates"
    value: List[CandidateEvent]
    type: Literal["list"] = "list"

    class Config:
        title = "Candidate Events"


# =====================================================================================
# Outputs
# =====================================================================================

class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[ShelfGuardPerson]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"
        json_schema_extra = {
            "outputStructure": [
                {
                    "boundingBox": {"left": "float", "top": "float", "width": "float", "height": "float"},
                    "confidence": "float",
                    "classLabel": "string",
                    "classId": "int",
                    "trackerID": "int",
                    "imgUID": "string",
                    "keyPoints": [{"cx": "float", "cy": "float", "confidence": "float"}],
                    "zones": ["string"],
                    "handState": "string",
                    "suspicionScore": "float",
                    "eventState": "string"
                }
            ],
            "outputNotes": [
                "Input pose detections are passed through and enriched with stage-1 state.",
                "zones: zone names the person's wrists are currently in (shelf / checkout / exit).",
                "handState: idle | inShelf | withProduct | inConcealment.",
                "eventState: normal | candidate | verifying | alarm."
            ]
        }


class OutputROI(Output):
    name: Literal["outputROI"] = "outputROI"
    value: List[ROI]
    type: Literal["list"] = "list"

    class Config:
        title = "Zones"


class OutputCandidates(Output):
    name: Literal["outputCandidates"] = "outputCandidates"
    value: List[CandidateEvent]
    type: Literal["list"] = "list"

    class Config:
        title = "Candidate Events"
        json_schema_extra = {
            "outputStructure": [
                {
                    "eventID": "string",
                    "trackerID": "int",
                    "startTime": "float",
                    "endTime": "float",
                    "score": "float",
                    "interactionSequence": [
                        {"phase": "string", "hand": "string", "startTime": "float", "endTime": "float"}
                    ],
                    "structuralSummary": "string",
                    "boundingBox": {"left": "float", "top": "float", "width": "float", "height": "float"},
                    "clipFrameKeys": ["string"],
                    "imgUID": "string"
                }
            ],
            "outputNotes": [
                "Emitted only when the temporal classifier score exceeds the candidate threshold.",
                "clipFrameKeys point to the sampled, anonymised clip frames stored in Redis.",
                "structuralSummary is the text added to the VLM prompt, e.g. 'hand in shelf 3.1-3.8 s, waist 4.2 s'."
            ]
        }


class OutputAlarms(Output):
    name: Literal["outputAlarms"] = "outputAlarms"
    value: List[TheftAlarm]
    type: Literal["list"] = "list"

    class Config:
        title = "Alarms"
        json_schema_extra = {
            "outputStructure": [
                {
                    "eventID": "string",
                    "trackerID": "int",
                    "decision": "theft | suspicious | normal",
                    "confidence": "float",
                    "evidenceStart": "float",
                    "evidenceEnd": "float",
                    "reason": "string",
                    "language": "string",
                    "notify": "bool",
                    "stage1Score": "float",
                    "latencyMs": "float",
                    "boundingBox": {"left": "float", "top": "float", "width": "float", "height": "float"},
                    "clipFrameKeys": ["string"],
                    "imgUID": "string"
                }
            ],
            "outputNotes": [
                "One entry per verified candidate; decisions not selected in 'Alarm Decisions' get notify=false.",
                "In Observation mode every entry has notify=false (shadow / field-validation mode).",
                "reason is the VLM justification, e.g. 'Kişi raftan aldığı ürünü ceket iç cebine yerleştirdi'."
            ]
        }


# =====================================================================================
# Config params - Zones
# =====================================================================================

class ConfigShelfZone(Config):
    """
        Draw the shelf areas once per camera.
        A wrist keypoint entering these polygons starts the 'hand in shelf' phase.
    """
    name: Literal["configShelfZone"] = "configShelfZone"
    value: str
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        title = "Shelf Zone"
        json_schema_extra = {
            "shortDescription": "Draw Shelf Areas",
            "class": "\\novavision\\app\\widgets\\ROI",
            "options": {
                "isMultiple": "true",
                "name": "shelfZone",
                "availableTypes": ["polyLine", "box"]
            }
        }


class ConfigCheckoutZone(Config):
    """
        Draw the checkout (cashier) areas.
        A person who reaches the checkout after picking a product is treated as legitimate behaviour.
    """
    name: Literal["configCheckoutZone"] = "configCheckoutZone"
    value: str
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        title = "Checkout Zone"
        json_schema_extra = {
            "shortDescription": "Draw Checkout Areas",
            "class": "\\novavision\\app\\widgets\\ROI",
            "options": {
                "isMultiple": "true",
                "name": "checkoutZone",
                "availableTypes": ["polyLine", "box"]
            }
        }


class ConfigExitZone(Config):
    """
        Draw the store exit areas.
        Moving towards the exit after a concealment phase without visiting the checkout raises the score.
    """
    name: Literal["configExitZone"] = "configExitZone"
    value: str
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        title = "Exit Zone"
        json_schema_extra = {
            "shortDescription": "Draw Exit Areas",
            "class": "\\novavision\\app\\widgets\\ROI",
            "options": {
                "isMultiple": "true",
                "name": "exitZone",
                "availableTypes": ["polyLine", "box"]
            }
        }


class ConcealmentWaist(Config):
    name: Literal["concealmentWaist"] = "concealmentWaist"
    value: Literal["Waist"] = "Waist"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Waist / Trouser Pocket"


class ConcealmentChest(Config):
    name: Literal["concealmentChest"] = "concealmentChest"
    value: Literal["Chest"] = "Chest"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Chest / Inner Jacket"


class ConcealmentBag(Config):
    name: Literal["concealmentBag"] = "concealmentBag"
    value: Literal["Bag"] = "Bag"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Personal Bag"


class ConfigConcealmentRegions(Config):
    """
        Body regions considered as concealment areas. They are derived from pose keypoints
        (hips -> waist, shoulders/hips -> chest) and from 'bag' detections.
    """
    name: Literal["configConcealmentRegions"] = "configConcealmentRegions"
    value: List[Union[ConcealmentWaist, ConcealmentChest, ConcealmentBag]]
    type: Literal["object"] = "object"
    field: Literal["selectBox"] = "selectBox"

    class Config:
        title = "Concealment Regions"
        json_schema_extra = {
            "shortDescription": "Body Regions Used for Concealment"
        }


# =====================================================================================
# Config params - Stage 1 (interaction + temporal classifier)
# =====================================================================================

class ConfigProductClasses(Config):
    """
        Comma separated class labels (from inputObjects) treated as products.
        Leave empty if no object detector is connected; hand-in-shelf is then used as the product cue.
    """
    name: Literal["configProductClasses"] = "configProductClasses"
    value: str = Field(default="product")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["e.g., product, bottle, cosmetic"] = "e.g., product, bottle, cosmetic"

    class Config:
        title = "Product Classes"
        json_schema_extra = {"shortDescription": "Product Class Labels"}


class ConfigBagClasses(Config):
    """
        Comma separated class labels treated as personal bags (concealment target).
    """
    name: Literal["configBagClasses"] = "configBagClasses"
    value: str = Field(default="bag, backpack, handbag")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["e.g., bag, backpack, handbag"] = "e.g., bag, backpack, handbag"

    class Config:
        title = "Bag Classes"
        json_schema_extra = {"shortDescription": "Personal Bag Class Labels"}


class ConfigBasketClasses(Config):
    """
        Comma separated class labels treated as store baskets / carts.
        Putting a product into these is legitimate and acts as a hard negative.
    """
    name: Literal["configBasketClasses"] = "configBasketClasses"
    value: str = Field(default="basket, cart")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["e.g., basket, cart"] = "e.g., basket, cart"

    class Config:
        title = "Basket / Cart Classes"
        json_schema_extra = {"shortDescription": "Legitimate Container Labels"}


class ConfigKeyPointConfidence(Config):
    """
        Minimum keypoint confidence for wrist / hip / shoulder points to be used in zone mapping.
    """
    name: Literal["configKeyPointConfidence"] = "configKeyPointConfidence"
    value: float = Field(default=0.3, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "KeyPoint Confidence"
        json_schema_extra = {"shortDescription": "Min Keypoint Confidence"}


class ConfigHandRadius(Config):
    """
        Radius of the hand area around a wrist keypoint, as a ratio of the person's bounding box height.
        Used to decide whether a hand touches a zone, product or concealment region.
    """
    name: Literal["configHandRadius"] = "configHandRadius"
    value: float = Field(default=0.06, ge=0.01, le=0.5)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.01, 0.5]"] = "[0.01, 0.5]"

    class Config:
        title = "Hand Radius"
        json_schema_extra = {"shortDescription": "Hand Area (ratio of body height)"}


class ConfigMaxPhaseGap(Config):
    """
        Maximum time (seconds) allowed between two consecutive interaction phases
        (shelf -> product -> concealment) for them to be chained into one sequence.
    """
    name: Literal["configMaxPhaseGap"] = "configMaxPhaseGap"
    value: float = Field(default=2.0, ge=0.1, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.1, 10.0]"] = "[0.1, 10.0]"

    class Config:
        title = "Max Phase Gap (s)"
        json_schema_extra = {"shortDescription": "Max Time Between Phases"}


class ConfigTemporalModelFile(Config):
    """
        Trained temporal classifier weights (ONNX / TensorRT engine).
    """
    name: Literal["configTemporalModelFile"] = "configTemporalModelFile"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"
    restart: Literal[True] = True

    class Config:
        title = "Model File"
        json_schema_extra = {
            "shortDescription": "Temporal Classifier Weights",
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": ["name"],
                "name": "temporalModelFile",
                "fileExtensions": ["onnx", "engine"],
            },
        }


class ConfigTemporalDeviceGPU(Config):
    name: Literal["configTemporalDeviceGPU"] = "configTemporalDeviceGPU"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigTemporalDeviceCPU(Config):
    name: Literal["configTemporalDeviceCPU"] = "configTemporalDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigTemporalDevice(Config):
    """
        Device used by the temporal classifier.
    """
    name: Literal["configTemporalDevice"] = "configTemporalDevice"
    value: Union[ConfigTemporalDeviceGPU, ConfigTemporalDeviceCPU]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"
        json_schema_extra = {"shortDescription": "Processing Device"}


class TemporalModelRuleBased(Config):
    configMaxPhaseGap: ConfigMaxPhaseGap
    name: Literal["temporalModelRuleBased"] = "temporalModelRuleBased"
    value: Literal["RuleBased"] = "RuleBased"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Rule Based (Baseline)"


class TemporalModelSTGCN(Config):
    configTemporalModelFile: ConfigTemporalModelFile
    configTemporalDevice: ConfigTemporalDevice
    name: Literal["temporalModelSTGCN"] = "temporalModelSTGCN"
    value: Literal["STGCN"] = "STGCN"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "ST-GCN (Skeleton)"


class TemporalModelTransformer(Config):
    configTemporalModelFile: ConfigTemporalModelFile
    configTemporalDevice: ConfigTemporalDevice
    name: Literal["temporalModelTransformer"] = "temporalModelTransformer"
    value: Literal["Transformer"] = "Transformer"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Transformer Encoder"


class ConfigTemporalModel(Config):
    """
        Temporal interaction classifier that scores each tracked person's sliding window.
        'Rule Based' chains the shelf -> product -> concealment phases without a trained model (baseline).
        'ST-GCN' and 'Transformer' use trained weights on pose sequences + interaction features.
    """
    name: Literal["configTemporalModel"] = "configTemporalModel"
    value: Union[TemporalModelRuleBased, TemporalModelSTGCN, TemporalModelTransformer]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Temporal Model"
        json_schema_extra = {"shortDescription": "Stage-1 Classifier"}


class ConfigWindowSize(Config):
    """
        Length (seconds) of the sliding window given to the temporal classifier. Recommended 2-4 s.
    """
    name: Literal["configWindowSize"] = "configWindowSize"
    value: float = Field(default=3.0, ge=1.0, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[1.0, 10.0]"] = "[1.0, 10.0]"

    class Config:
        title = "Window Size (s)"
        json_schema_extra = {"shortDescription": "Sliding Window Length"}


class ConfigWindowStride(Config):
    """
        Step (seconds) between two consecutive sliding windows. Smaller values react faster but cost more.
    """
    name: Literal["configWindowStride"] = "configWindowStride"
    value: float = Field(default=0.5, ge=0.1, le=5.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.1, 5.0]"] = "[0.1, 5.0]"

    class Config:
        title = "Window Stride (s)"
        json_schema_extra = {"shortDescription": "Sliding Window Step"}


class ConfigCandidateThreshold(Config):
    """
        Minimum stage-1 score for a window to become a candidate event.
        Keep it low (high recall, target >= 0.90); precision is the VLM's job.
    """
    name: Literal["configCandidateThreshold"] = "configCandidateThreshold"
    value: float = Field(default=0.3, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Candidate Threshold"
        json_schema_extra = {"shortDescription": "Stage-1 Score Threshold"}


class ConfigMinTrackDuration(Config):
    """
        Minimum time (seconds) a person must be tracked before being evaluated. Filters out ID switches.
    """
    name: Literal["configMinTrackDuration"] = "configMinTrackDuration"
    value: float = Field(default=1.0, ge=0.0, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 10.0]"] = "[0.0, 10.0]"

    class Config:
        title = "Min Track Duration (s)"
        json_schema_extra = {"shortDescription": "Min Tracking Time"}


class ConfigEventCooldown(Config):
    """
        After a candidate is produced for a tracker ID, new candidates for the same ID are
        suppressed for this many seconds to avoid duplicate events.
    """
    name: Literal["configEventCooldown"] = "configEventCooldown"
    value: float = Field(default=10.0, ge=0.0, le=300.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 300.0]"] = "[0.0, 300.0]"

    class Config:
        title = "Event Cooldown (s)"
        json_schema_extra = {"shortDescription": "Duplicate Event Suppression"}


# =====================================================================================
# Config params - Clip extraction & anonymisation
# =====================================================================================

class ConfigNumFrames(Config):
    """
        Number of frames uniformly sampled from the candidate clip and sent to the VLM. Recommended 8-16.
    """
    name: Literal["configNumFrames"] = "configNumFrames"
    value: int = Field(default=8, ge=2, le=32)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, 32]"] = "[2, 32]"

    class Config:
        title = "Frames per Clip"
        json_schema_extra = {"shortDescription": "Sampled Frames for VLM"}


class ConfigClipPreSeconds(Config):
    """
        Seconds of video kept before the candidate window starts.
    """
    name: Literal["configClipPreSeconds"] = "configClipPreSeconds"
    value: float = Field(default=1.0, ge=0.0, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 10.0]"] = "[0.0, 10.0]"

    class Config:
        title = "Pre-Event (s)"
        json_schema_extra = {"shortDescription": "Seconds Before Event"}


class ConfigClipPostSeconds(Config):
    """
        Seconds of video kept after the candidate window ends.
    """
    name: Literal["configClipPostSeconds"] = "configClipPostSeconds"
    value: float = Field(default=1.0, ge=0.0, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 10.0]"] = "[0.0, 10.0]"

    class Config:
        title = "Post-Event (s)"
        json_schema_extra = {"shortDescription": "Seconds After Event"}


class ConfigBlurStrength(Config):
    """
        Strength of the face anonymisation (kernel size ratio of the face box).
    """
    name: Literal["configBlurStrength"] = "configBlurStrength"
    value: float = Field(default=0.5, ge=0.1, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.1, 1.0]"] = "[0.1, 1.0]"

    class Config:
        title = "Strength"
        json_schema_extra = {"shortDescription": "Anonymisation Strength"}


class AnonymizeMethodGaussian(Config):
    name: Literal["anonymizeMethodGaussian"] = "anonymizeMethodGaussian"
    value: Literal["Gaussian"] = "Gaussian"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Gaussian Blur"


class AnonymizeMethodPixelate(Config):
    name: Literal["anonymizeMethodPixelate"] = "anonymizeMethodPixelate"
    value: Literal["Pixelate"] = "Pixelate"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Pixelate"


class ConfigAnonymizeMethod(Config):
    """
        Face anonymisation method.
    """
    name: Literal["configAnonymizeMethod"] = "configAnonymizeMethod"
    value: Union[AnonymizeMethodGaussian, AnonymizeMethodPixelate]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Method"
        json_schema_extra = {"shortDescription": "Anonymisation Method"}


class AnonymizeEnable(Config):
    configAnonymizeMethod: ConfigAnonymizeMethod
    configBlurStrength: ConfigBlurStrength
    name: Literal["anonymizeEnable"] = "anonymizeEnable"
    value: Literal["Enable"] = "Enable"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class AnonymizeDisable(Config):
    name: Literal["anonymizeDisable"] = "anonymizeDisable"
    value: Literal["Disable"] = "Disable"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigAnonymize(Config):
    """
        Blurs faces in the extracted clip before it leaves the device (KVKK).
        Face area is estimated from pose head keypoints (nose, eyes, ears); no face recognition is used.
    """
    name: Literal["configAnonymize"] = "configAnonymize"
    value: Union[AnonymizeEnable, AnonymizeDisable]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Face Anonymisation"
        json_schema_extra = {"shortDescription": "Blur Faces in Clip (KVKK)"}


class HighlightSuspectTrue(Config):
    name: Literal["highlightSuspectTrue"] = "highlightSuspectTrue"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class HighlightSuspectFalse(Config):
    name: Literal["highlightSuspectFalse"] = "highlightSuspectFalse"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHighlightSuspect(Config):
    """
        Draws the suspect's bounding box on the clip frames sent to the VLM (ablation parameter).
    """
    name: Literal["configHighlightSuspect"] = "configHighlightSuspect"
    value: Union[HighlightSuspectTrue, HighlightSuspectFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Highlight Suspect"
        json_schema_extra = {"shortDescription": "Draw Suspect Box on Clip"}


# =====================================================================================
# Config params - Stage 2 (VLM)
# =====================================================================================

class VlmModelQwen2VL2B(Config):
    name: Literal["vlmModelQwen2VL2B"] = "vlmModelQwen2VL2B"
    value: Literal["Qwen/Qwen2-VL-2B-Instruct"] = "Qwen/Qwen2-VL-2B-Instruct"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen2-VL 2B"


class VlmModelQwen2VL7B(Config):
    name: Literal["vlmModelQwen2VL7B"] = "vlmModelQwen2VL7B"
    value: Literal["Qwen/Qwen2-VL-7B-Instruct"] = "Qwen/Qwen2-VL-7B-Instruct"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen2-VL 7B"


class VlmModelQwen25VL3B(Config):
    name: Literal["vlmModelQwen25VL3B"] = "vlmModelQwen25VL3B"
    value: Literal["Qwen/Qwen2.5-VL-3B-Instruct"] = "Qwen/Qwen2.5-VL-3B-Instruct"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen2.5-VL 3B"


class VlmModelVideoLLaVA7B(Config):
    name: Literal["vlmModelVideoLLaVA7B"] = "vlmModelVideoLLaVA7B"
    value: Literal["LanguageBind/Video-LLaVA-7B-hf"] = "LanguageBind/Video-LLaVA-7B-hf"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Video-LLaVA 7B"


class ConfigVlmModel(Config):
    """
        Open-weight 2-8B parameter VLM used for verification.
    """
    name: Literal["configVlmModel"] = "configVlmModel"
    value: Union[VlmModelQwen2VL2B, VlmModelQwen2VL7B, VlmModelQwen25VL3B, VlmModelVideoLLaVA7B]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Base Model"
        json_schema_extra = {"shortDescription": "VLM Base Model"}


class ConfigLoraAdapterFile(Config):
    """
        LoRA adapter (fine-tuned on labelled store clips) applied on top of the base model.
    """
    name: Literal["configLoraAdapterFile"] = "configLoraAdapterFile"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"
    restart: Literal[True] = True

    class Config:
        title = "Adapter File"
        json_schema_extra = {
            "shortDescription": "LoRA Adapter (.zip / .safetensors)",
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": ["name"],
                "name": "loraAdapterFile",
                "fileExtensions": ["zip", "safetensors"],
            },
        }


class LoraEnable(Config):
    configLoraAdapterFile: ConfigLoraAdapterFile
    name: Literal["loraEnable"] = "loraEnable"
    value: Literal["Enable"] = "Enable"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class LoraDisable(Config):
    name: Literal["loraDisable"] = "loraDisable"
    value: Literal["Disable"] = "Disable"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable (Zero-Shot)"


class ConfigLora(Config):
    """
        Use a LoRA fine-tuned adapter or run the base model zero-shot.
    """
    name: Literal["configLora"] = "configLora"
    value: Union[LoraDisable, LoraEnable]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "LoRA Adapter"
        json_schema_extra = {"shortDescription": "Fine-Tuned Adapter"}


class Quantization4Bit(Config):
    name: Literal["quantization4Bit"] = "quantization4Bit"
    value: Literal["4bit"] = "4bit"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "4-bit"


class Quantization8Bit(Config):
    name: Literal["quantization8Bit"] = "quantization8Bit"
    value: Literal["8bit"] = "8bit"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "8-bit"


class QuantizationFP16(Config):
    name: Literal["quantizationFP16"] = "quantizationFP16"
    value: Literal["fp16"] = "fp16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FP16 (No Quantization)"


class ConfigQuantization(Config):
    """
        Weight precision of the local VLM. 4-bit fits 7B models on a single 24 GB GPU.
    """
    name: Literal["configQuantization"] = "configQuantization"
    value: Union[Quantization4Bit, Quantization8Bit, QuantizationFP16]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Quantization"
        json_schema_extra = {"shortDescription": "Model Precision"}


class ConfigVlmDeviceGPU(Config):
    name: Literal["configVlmDeviceGPU"] = "configVlmDeviceGPU"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigVlmDeviceCPU(Config):
    name: Literal["configVlmDeviceCPU"] = "configVlmDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigVlmDevice(Config):
    """
        Device used by the local VLM. CPU is only meant for testing.
    """
    name: Literal["configVlmDevice"] = "configVlmDevice"
    value: Union[ConfigVlmDeviceGPU, ConfigVlmDeviceCPU]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"
        json_schema_extra = {"shortDescription": "Processing Device"}


class ConfigVlmEndpoint(Config):
    """
        OpenAI-compatible endpoint of the VLM inference server (e.g. vLLM) running in the store.
    """
    name: Literal["configVlmEndpoint"] = "configVlmEndpoint"
    value: str = Field(default="http://127.0.0.1:8001/v1")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["http://host:port/v1"] = "http://host:port/v1"
    restart: Literal[True] = True

    class Config:
        title = "Endpoint URL"
        json_schema_extra = {"shortDescription": "VLM Server URL"}


class ConfigVlmServedModel(Config):
    """
        Model name served by the inference server (as passed to the 'model' field of the request).
    """
    name: Literal["configVlmServedModel"] = "configVlmServedModel"
    value: str = Field(default="shelfguard-vlm")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["e.g., shelfguard-vlm"] = "e.g., shelfguard-vlm"
    restart: Literal[True] = True

    class Config:
        title = "Served Model Name"
        json_schema_extra = {"shortDescription": "Model Name on Server"}


class ConfigVlmApiKey(Config):
    """
        API key of the inference server. Leave empty if the server does not require one.
    """
    name: Literal["configVlmApiKey"] = "configVlmApiKey"
    value: str = Field(default="")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    restart: Literal[True] = True

    class Config:
        title = "API Key"
        json_schema_extra = {"shortDescription": "Server API Key (optional)"}


class VlmBackendLocal(Config):
    configVlmModel: ConfigVlmModel
    configLora: ConfigLora
    configQuantization: ConfigQuantization
    configVlmDevice: ConfigVlmDevice
    name: Literal["vlmBackendLocal"] = "vlmBackendLocal"
    value: Literal["Local"] = "Local"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Local (In-Process)"


class VlmBackendServer(Config):
    configVlmEndpoint: ConfigVlmEndpoint
    configVlmServedModel: ConfigVlmServedModel
    configVlmApiKey: ConfigVlmApiKey
    name: Literal["vlmBackendServer"] = "vlmBackendServer"
    value: Literal["Server"] = "Server"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Inference Server (vLLM)"


class ConfigVlmBackend(Config):
    """
        Where the VLM runs.
        'Local' loads the model inside this capsule (edge device / GPU of this server).
        'Inference Server' sends the clip to an OpenAI-compatible server (e.g. vLLM) in the store.
    """
    name: Literal["configVlmBackend"] = "configVlmBackend"
    value: Union[VlmBackendLocal, VlmBackendServer]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "VLM Backend"
        json_schema_extra = {"shortDescription": "VLM Runtime"}


class LanguageTurkish(Config):
    name: Literal["languageTurkish"] = "languageTurkish"
    value: Literal["tr"] = "tr"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Turkish"


class LanguageEnglish(Config):
    name: Literal["languageEnglish"] = "languageEnglish"
    value: Literal["en"] = "en"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "English"


class LanguageEnglishToTurkish(Config):
    name: Literal["languageEnglishToTurkish"] = "languageEnglishToTurkish"
    value: Literal["en-tr"] = "en-tr"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "English + Translate to Turkish"


class ConfigOutputLanguage(Config):
    """
        Language of the VLM justification.
        'English + Translate' asks the model in English and translates the reason with a light translation step
        (fallback when native Turkish output quality is low).
    """
    name: Literal["configOutputLanguage"] = "configOutputLanguage"
    value: Union[LanguageTurkish, LanguageEnglish, LanguageEnglishToTurkish]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Reason Language"
        json_schema_extra = {"shortDescription": "Justification Language"}


class StructuralSummaryTrue(Config):
    name: Literal["structuralSummaryTrue"] = "structuralSummaryTrue"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class StructuralSummaryFalse(Config):
    name: Literal["structuralSummaryFalse"] = "structuralSummaryFalse"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigStructuralSummary(Config):
    """
        Adds the stage-1 structural summary (e.g. 'hand in shelf 3.1-3.8 s, waist 4.2 s') to the prompt (ablation parameter).
    """
    name: Literal["configStructuralSummary"] = "configStructuralSummary"
    value: Union[StructuralSummaryTrue, StructuralSummaryFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Structural Summary in Prompt"
        json_schema_extra = {"shortDescription": "Add Stage-1 Summary to Prompt"}


class ConfigPromptInstruction(Config):
    """
        Optional extra instruction appended to the built-in prompt (store specific rules etc.).
        The model is always asked to return JSON: decision, confidence, evidence interval and reason.
    """
    name: Literal["configPromptInstruction"] = "configPromptInstruction"
    value: str = Field(default="")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["e.g., Cosmetics aisle, small items are often hidden in sleeves."] = \
        "e.g., Cosmetics aisle, small items are often hidden in sleeves."

    class Config:
        title = "Extra Prompt Instruction"
        json_schema_extra = {"shortDescription": "Additional Prompt Text"}


class ConfigMaxNewTokens(Config):
    """
        Maximum number of tokens the VLM may generate. Lower values reduce latency.
    """
    name: Literal["configMaxNewTokens"] = "configMaxNewTokens"
    value: int = Field(default=256, ge=32, le=1024)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[32, 1024]"] = "[32, 1024]"

    class Config:
        title = "Max New Tokens"
        json_schema_extra = {"shortDescription": "Generation Length Limit"}


class ConfigTemperature(Config):
    """
        Sampling temperature. 0 gives deterministic, reproducible decisions.
    """
    name: Literal["configTemperature"] = "configTemperature"
    value: float = Field(default=0.0, ge=0.0, le=2.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 2.0]"] = "[0.0, 2.0]"

    class Config:
        title = "Temperature"
        json_schema_extra = {"shortDescription": "Sampling Temperature"}


class ConfigVlmTimeout(Config):
    """
        Maximum time (seconds) to wait for a VLM answer. On timeout the candidate is reported
        as 'suspicious' with the stage-1 score so no event is lost. Target end-to-end latency is < 5 s.
    """
    name: Literal["configVlmTimeout"] = "configVlmTimeout"
    value: float = Field(default=5.0, ge=1.0, le=60.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[1.0, 60.0]"] = "[1.0, 60.0]"

    class Config:
        title = "VLM Timeout (s)"
        json_schema_extra = {"shortDescription": "Max VLM Wait Time"}


# =====================================================================================
# Config params - Alarm
# =====================================================================================

class AlarmDecisionTheft(Config):
    name: Literal["alarmDecisionTheft"] = "alarmDecisionTheft"
    value: Literal["theft"] = "theft"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Theft"


class AlarmDecisionSuspicious(Config):
    name: Literal["alarmDecisionSuspicious"] = "alarmDecisionSuspicious"
    value: Literal["suspicious"] = "suspicious"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Suspicious"


class ConfigAlarmDecisions(Config):
    """
        VLM decisions that raise an alarm (notify=true). 'normal' never raises an alarm.
    """
    name: Literal["configAlarmDecisions"] = "configAlarmDecisions"
    value: List[Union[AlarmDecisionTheft, AlarmDecisionSuspicious]]
    type: Literal["object"] = "object"
    field: Literal["selectBox"] = "selectBox"

    class Config:
        title = "Alarm Decisions"
        json_schema_extra = {"shortDescription": "Decisions That Raise an Alarm"}


class ConfigAlarmConfidence(Config):
    """
        Minimum VLM confidence for an alarm to be raised.
    """
    name: Literal["configAlarmConfidence"] = "configAlarmConfidence"
    value: float = Field(default=0.5, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Alarm Confidence"
        json_schema_extra = {"shortDescription": "Min VLM Confidence"}


class OperationModeActive(Config):
    name: Literal["operationModeActive"] = "operationModeActive"
    value: Literal["Active"] = "Active"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Active (Notify Staff)"


class OperationModeObservation(Config):
    name: Literal["operationModeObservation"] = "operationModeObservation"
    value: Literal["Observation"] = "Observation"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Observation Only"


class ConfigOperationMode(Config):
    """
        'Active' sends alarms to staff. 'Observation Only' logs every result with notify=false
        (used for the 4-week shadow field validation).
    """
    name: Literal["configOperationMode"] = "configOperationMode"
    value: Union[OperationModeActive, OperationModeObservation]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Operation Mode"
        json_schema_extra = {"shortDescription": "Alarm or Shadow Mode"}


class DrawOverlayTrue(Config):
    name: Literal["drawOverlayTrue"] = "drawOverlayTrue"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class DrawOverlayFalse(Config):
    name: Literal["drawOverlayFalse"] = "drawOverlayFalse"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigDrawOverlay(Config):
    """
        Draws zones, skeletons, hand states and alarm labels on the output image.
    """
    name: Literal["configDrawOverlay"] = "configDrawOverlay"
    value: Union[DrawOverlayTrue, DrawOverlayFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Draw Overlay"
        json_schema_extra = {"shortDescription": "Visualise on Output Image"}


# =====================================================================================
# Executor: ShelfGuard (Hybrid: Stage 1 + Stage 2)
# =====================================================================================

class ShelfGuardInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections
    inputObjects: Optional[InputObjects]


class ShelfGuardConfigs(Configs):
    # Zones
    configShelfZone: ConfigShelfZone
    configCheckoutZone: ConfigCheckoutZone
    configExitZone: ConfigExitZone
    configConcealmentRegions: ConfigConcealmentRegions
    # Interaction / Stage 1
    configProductClasses: ConfigProductClasses
    configBagClasses: ConfigBagClasses
    configBasketClasses: ConfigBasketClasses
    configKeyPointConfidence: ConfigKeyPointConfidence
    configHandRadius: ConfigHandRadius
    configTemporalModel: ConfigTemporalModel
    configWindowSize: ConfigWindowSize
    configWindowStride: ConfigWindowStride
    configCandidateThreshold: ConfigCandidateThreshold
    configMinTrackDuration: ConfigMinTrackDuration
    configEventCooldown: ConfigEventCooldown
    # Clip
    configNumFrames: ConfigNumFrames
    configClipPreSeconds: ConfigClipPreSeconds
    configClipPostSeconds: ConfigClipPostSeconds
    configAnonymize: ConfigAnonymize
    configHighlightSuspect: ConfigHighlightSuspect
    # VLM / Stage 2
    configVlmBackend: ConfigVlmBackend
    configOutputLanguage: ConfigOutputLanguage
    configStructuralSummary: ConfigStructuralSummary
    configPromptInstruction: ConfigPromptInstruction
    configMaxNewTokens: ConfigMaxNewTokens
    configTemperature: ConfigTemperature
    configVlmTimeout: ConfigVlmTimeout
    # Alarm
    configAlarmDecisions: ConfigAlarmDecisions
    configAlarmConfidence: ConfigAlarmConfidence
    configOperationMode: ConfigOperationMode
    configDrawOverlay: ConfigDrawOverlay


class ShelfGuardOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections
    outputAlarms: OutputAlarms
    outputROI: OutputROI


class ShelfGuardRequest(Request):
    inputs: Optional[ShelfGuardInputs]
    configs: ShelfGuardConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ShelfGuardResponse(Response):
    outputs: ShelfGuardOutputs


class ShelfGuardExecutor(Config):
    name: Literal["ShelfGuard"] = "ShelfGuard"
    value: Union[ShelfGuardRequest, ShelfGuardResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Hybrid (Stage 1 + VLM)"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# =====================================================================================
# Executor: CandidateDetection (Stage 1 only - edge)
# =====================================================================================

class CandidateDetectionInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections
    inputObjects: Optional[InputObjects]


class CandidateDetectionConfigs(Configs):
    # Zones
    configShelfZone: ConfigShelfZone
    configCheckoutZone: ConfigCheckoutZone
    configExitZone: ConfigExitZone
    configConcealmentRegions: ConfigConcealmentRegions
    # Interaction / Stage 1
    configProductClasses: ConfigProductClasses
    configBagClasses: ConfigBagClasses
    configBasketClasses: ConfigBasketClasses
    configKeyPointConfidence: ConfigKeyPointConfidence
    configHandRadius: ConfigHandRadius
    configTemporalModel: ConfigTemporalModel
    configWindowSize: ConfigWindowSize
    configWindowStride: ConfigWindowStride
    configCandidateThreshold: ConfigCandidateThreshold
    configMinTrackDuration: ConfigMinTrackDuration
    configEventCooldown: ConfigEventCooldown
    # Clip
    configNumFrames: ConfigNumFrames
    configClipPreSeconds: ConfigClipPreSeconds
    configClipPostSeconds: ConfigClipPostSeconds
    configAnonymize: ConfigAnonymize
    configHighlightSuspect: ConfigHighlightSuspect
    # Visualisation
    configDrawOverlay: ConfigDrawOverlay


class CandidateDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections
    outputCandidates: OutputCandidates
    outputROI: OutputROI


class CandidateDetectionRequest(Request):
    inputs: Optional[CandidateDetectionInputs]
    configs: CandidateDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class CandidateDetectionResponse(Response):
    outputs: CandidateDetectionOutputs


class CandidateDetectionExecutor(Config):
    name: Literal["CandidateDetection"] = "CandidateDetection"
    value: Union[CandidateDetectionRequest, CandidateDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Stage 1 - Candidate Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# =====================================================================================
# Executor: VLMVerification (Stage 2 only - GPU server)
# =====================================================================================

class VLMVerificationInputs(Inputs):
    inputImage: InputImage
    inputCandidates: InputCandidates


class VLMVerificationConfigs(Configs):
    configVlmBackend: ConfigVlmBackend
    configOutputLanguage: ConfigOutputLanguage
    configStructuralSummary: ConfigStructuralSummary
    configPromptInstruction: ConfigPromptInstruction
    configMaxNewTokens: ConfigMaxNewTokens
    configTemperature: ConfigTemperature
    configVlmTimeout: ConfigVlmTimeout
    configAlarmDecisions: ConfigAlarmDecisions
    configAlarmConfidence: ConfigAlarmConfidence
    configOperationMode: ConfigOperationMode
    configDrawOverlay: ConfigDrawOverlay


class VLMVerificationOutputs(Outputs):
    outputImage: OutputImage
    outputAlarms: OutputAlarms


class VLMVerificationRequest(Request):
    inputs: Optional[VLMVerificationInputs]
    configs: VLMVerificationConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class VLMVerificationResponse(Response):
    outputs: VLMVerificationOutputs


class VLMVerificationExecutor(Config):
    name: Literal["VLMVerification"] = "VLMVerification"
    value: Union[VLMVerificationRequest, VLMVerificationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Stage 2 - VLM Verification"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# =====================================================================================
# Package
# =====================================================================================

class ConfigExecutor(Config):
    """
        Selects how the capsule runs:
        'Hybrid' runs the full cascade (stage 1 candidates verified by the VLM) in one node.
        'Stage 1' only produces candidate events (edge device, baseline evaluation).
        'Stage 2' only verifies candidate events with the VLM (separate GPU server).
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ShelfGuardExecutor, CandidateDetectionExecutor, VLMVerificationExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {
            "shortDescription": "Select Pipeline Stage"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["ShelfGuard"] = "ShelfGuard"
