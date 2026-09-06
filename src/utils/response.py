
from sdks.novavision.src.helper.package import PackageHelper
from components.Test.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, OutputImage, RotateExecutorOutputs, RotateExecutorResponse, RotateExecutor, ResizeExecutorOutputs, ResizeExecutorResponse, ResizeExecutor


def build_response_rotate(context):
    outputImage = OutputImage(value=context.image)
    outputs = RotateExecutorOutputs(outputImage=outputImage)
    packageResponse = RotateExecutorResponse(outputs=outputs)
    packageExecutor = RotateExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_resize(context):
    outputImage = OutputImage(value=context.image)
    outputs = ResizeExecutorOutputs(outputImage=outputImage)
    packageResponse = ResizeExecutorResponse(outputs=outputs)
    packageExecutor = ResizeExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
    