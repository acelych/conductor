from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='topk_gather',
    ext_modules=[
        CUDAExtension('topk_gather', [
            'topk_gather.cpp',
            'topk_gather.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })