import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from pybind11_stubgen import run

setup(
    name='cdt_extensions',
    ext_modules=[
        CUDAExtension(
            name='cdt_extensions', 
            sources=[
                'cdt_extensions.cpp',
                'cross_hada.cu',
                'topk.cu'
                ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '--expt-relaxed-constexpr']
            },
            # for PyTorch 2.1+
            export_types=True if hasattr(CUDAExtension, 'export_types') else False
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

def stub(module_name: str):
    import logging
    from pybind11_stubgen import CLIArgs, stub_parser_from_args, Printer, to_output_and_subdir, run, Writer
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - [%(levelname)7s] %(message)s",
    )
    args = CLIArgs(
        module_name=module_name,
        output_dir='./',
        stub_extension="pyi",
        # default ags:
        root_suffix=None,
        ignore_all_errors=False,
        ignore_invalid_identifiers=None,
        ignore_invalid_expressions=None,
        ignore_unresolved_names=None,
        exit_code=False,
        numpy_array_wrap_with_annotated=False,
        numpy_array_use_type_var=False,
        numpy_array_remove_parameters=False,
        enum_class_locations=[],
        print_safe_value_reprs=None,
        print_invalid_expressions_as_is=False,
        dry_run=False
    )

    parser = stub_parser_from_args(args)
    printer = Printer(invalid_expr_as_ellipses=not args.print_invalid_expressions_as_is)

    out_dir, sub_dir = to_output_and_subdir(
        output_dir=args.output_dir,
        module_name=args.module_name,
        root_suffix=args.root_suffix,
    )

    run(
        parser,
        printer,
        args.module_name,
        out_dir,
        sub_dir=sub_dir,
        dry_run=args.dry_run,
        writer=Writer(stub_ext=args.stub_extension),
    )
    
    print("pybind11_stubgen: .pyi generated.")

stub('cdt_extensions')