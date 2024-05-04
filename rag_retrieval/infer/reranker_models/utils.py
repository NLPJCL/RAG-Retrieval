from typing import Union, Optional, List, Iterable


def vprint(txt: str, verbose: int) -> None:
    if verbose > 0:
        print(txt)


try:
    import torch
    from transformers  import is_torch_npu_available

    def get_dtype(
        dtype: str,
        device: str,
        verbose: int = 1,
    ) -> torch.dtype:
        if dtype is None:
            vprint("No dtype set", verbose)
        print(dtype)
        if device == "cpu":
            vprint("Device set to `cpu`, setting dtype to `float32`", verbose)
            dtype = torch.float32
            return dtype
        elif dtype == "fp16" or dtype == "float16":
            dtype = torch.float16
        elif dtype == "bf16" or dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        vprint(f"Using dtype {dtype}", verbose)
        return dtype
    
    def get_device(
        device: str ,
        verbose: int = 1,
        no_mps: bool = False,
    ) -> Union[str, torch.device]:
        if not device:
            vprint("No device set", verbose)
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            elif is_torch_npu_available():
                device = torch.device("npu")
            else:
                device = "cpu"
        vprint(f"Using device {device}", verbose)
        return device

except ImportError:
    print("Torch or transformers not installed...")