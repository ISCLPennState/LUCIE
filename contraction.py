import torch


@torch.jit.script
def compl_mul1d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 1-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd1d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 1-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul1d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def compl_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 2-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 2-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script  # TODO remove
def _contract_localconv_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex local convolution operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script  # TODO remove
def _contract_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex block convolution operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bim,imn->bin", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script  # TODO remove
def _contractadd_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex block convolution operation between two tensors 'a' and 'b', and
    then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(_contract_blockconv_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# for the experimental layer
@torch.jit.script  # TODO remove
def compl_exp_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_exp_muladd2d_fwd(  # TODO remove
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b',
    and then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_exp_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def real_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D real multiplication operation between two tensors 'a' and 'b'.
    """
    res = torch.einsum("bixy,io->boxy", a, b)
    return res


@torch.jit.script
def real_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D real multiplication operation between two tensors 'a' and 'b', and
    then adds a third tensor 'c'.
    """
    res = real_mul2d_fwd(a, b) + c
    return res


# new contractions set to replace older ones. We use complex
@torch.jit.script
def _contract_diagonal(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex diagonal operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ioxy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_dhconv(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex Driscoll-Healy style convolution operation between two tensors
    'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_diagonal(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex convolution operation between two tensors 'a' and 'b'
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ixy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_dhconv(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex convolution operation between two tensors 'a' and 'b'
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ix->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res

