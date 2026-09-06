"""
    It is one of the preprocessing components in which the image is Resize.
"""

import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image #media/
from sdks.novavision.src.base.component import Component # casule/ vb.
from sdks.novavision.src.helper.executor import Executor # Executoru 
from components.Package.src.utils.response import build_response_resize # gerekli
from components.Package.src.models.PackageModel import PackageModel #gerekli


class ResizeExecutor(Component): # Sabit
    def __init__(self, request, bootstrap): # Sabit
        super().__init__(request, bootstrap) # Sabit
        self.request.model = PackageModel(**(self.request.data)) # Sabit
        self.resize 


    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}


    def run(self):
        pass


if "__main__" == __name__:
    Executor(sys.argv[1]).run()