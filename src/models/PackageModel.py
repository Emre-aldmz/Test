
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


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


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
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

class SizeImage(Config):
    """
        Resize for image
    """
    name: Literal["SizeImage"] = "SizeImage"
    value: int = Field(ge=32, le=4096,default=640) # input tipi veriliyor ve türü belirleniyor
    type: Literal["number"] = "number" # value muzun tipi (number, object, string, bool, list, dict)
    field: Literal["textInput"] = "textInput" # parametrenin web arayüzünde görünüşü 
    placeHolder: Literal["[32, 4096]"] = "[32, 4096]" # Alan doldurulmadan önce soluk renkle yazan yazı

    class Config:
        title = "Resize"
        json_schema_extra = {
            "shortDescription": "Resize Degree"
        }
 
class ResizeExecutorInputs(Inputs):
    inputImage: InputImage
           
class ResizeExecutorConfigs(Configs):
    sizeImage: SizeImage  

class ResizeExecutorOutputs(Outputs):
    outputImage: OutputImage

class ResizeExecutorRequest(Request):
    inputs: Optional[ResizeExecutorInputs]
    configs: ResizeExecutorConfigs

class ResizeExecutorResponse(Response):
    outputs: ResizeExecutorOutputs

class ResizeExecutor(Config):
    name: Literal["ResizeExecutor"] = "ResizeExecutor"
    value: Union[ResizeExecutorRequest, ResizeExecutorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Resize"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

# Rotate Executor

class KeepSideFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class KeepSideTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option" 

    class Config:
        title = "Enable"


class KeepSideBBox(Config):
    """
        Rotate image without catting off sides.
    """
    name: Literal["KeepSide"] = "KeepSide"
    value: Union[KeepSideTrue, KeepSideFalse] # union(birleştirme) birden fazla değer. ya o ya o
    type: Literal["object"] = "object" # object çünkü class
    field: Literal["dropdownlist"] = "dropdownlist" 

    class Config:
        title = "Keep Sides"


class Rotate(Config):
    """
        Positive angles specify counterclockwise rotation while negative angles indicate clockwise rotation. :)
    """
    name: Literal["Rotate"] = "Rotate"
    value: int = Field(ge=-359.0, le=359.0,default=90) # input tipi veriliyor ve türü belirleniyor
    type: Literal["number"] = "number" # value muzun tipi (number, object, string, bool, list, dict)
    field: Literal["textInput"] = "textInput" # parametrenin web arayüzünde görünüşü 
    placeHolder: Literal["[-359, 359]"] = "[-359, 359]" # Alan doldurulmadan önce soluk renkle yazan yazı

    class Config:
        title = "Rotate"
        json_schema_extra = {
            "shortDescription": "Rotate Degree"
        }


class RotateExecutorInputs(Inputs):
    inputImage: InputImage # İnputları belirle alt alta | baş harf küçük


class RotateExecutorConfigs(Configs):
    rotate: Rotate  
    drawBBox: KeepSideBBox


class RotateExecutorOutputs(Outputs):
    outputImage: OutputImage # Keylerin baş harfi her zaman küçük Çıktıları class olucak


class RotateExecutorRequest(Request):
    inputs: Optional[RotateExecutorInputs] # İnput alması zorunlu değil Optional
    configs: RotateExecutorConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }
    
class RotateExecutorResponse(Response):
    outputs: RotateExecutorOutputs

class RotateExecutor(Config):
    name: Literal["RotateExecutor"] = "RotateExecutor"
    value: Union[RotateExecutorRequest, RotateExecutorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Rotate"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

# General

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor" # değişmez
    value: Union[RotateExecutor, ResizeExecutor] #executorları yazma yeri.
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task" 


class PackageConfigs(Configs):
    executor: ConfigExecutor
    # Sistemsel

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["Test"] = "Test"

    # Literal değiştirilemez!!