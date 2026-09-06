
from sdks.novavision.src.helper.package import PackageHelper
from components.Test.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, OutputImage, RotateExecutorOutputs, RotateExecutorResponse, RotateExecutor, ResizeExecutorOutputs, ResizeExecutorResponse, ResizeExecutor


def build_response_rotate(context):
    outputImage = OutputImage(value=context.image)
    outputs = RotateExecutorOutputs(outputImage=outputImage)
    RotateExecutorResponse = RotateExecutorResponse(outputs=outputs)
    RotateExecutor = RotateExecutor(value=RotateExecutorResponse)
    executor = ConfigExecutor(value=RotateExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_resize(context):
    outputImage = OutputImage(value=context.image)
    outputs = ResizeExecutorOutputs(outputImage=outputImage)
    ResizeExecutorResponse = ResizeExecutorResponse(outputs=outputs)
    ResizeExecutor = ResizeExecutor(value=ResizeExecutorResponse)
    executor = ConfigExecutor(value=ResizeExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
    