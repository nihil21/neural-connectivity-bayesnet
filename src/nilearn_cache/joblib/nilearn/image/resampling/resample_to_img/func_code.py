# first line: 612
def resample_to_img(source_img, target_img,
                    interpolation='continuous', copy=True, order='F',
                    clip=False, fill_value=0, force_resample=False):
    """Resample a Niimg-like source image on a target Niimg-like image
    (no registration is performed: the image should already be aligned).

    .. versionadded:: 0.2.4

    Parameters
    ----------
    source_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Image(s) to resample.

    target_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Reference image taken for resampling.

    interpolation : str, optional
        Can be 'continuous', 'linear', or 'nearest'. Indicates the resample
        method. Default='continuous'.

    copy : bool, optional
        If True, guarantees that output array has no memory in common with
        input array.
        In all cases, input images are never modified by this function.
        Default=True.

    order : "F" or "C", optional
        Data ordering in output array. This function is slightly faster with
        Fortran ordering. Default="F".

    clip : bool, optional
        If False (default) no clip is preformed.
        If True all resampled image values above max(img) and under min(img) are
        clipped to min(img) and max(img). Default=False.

    fill_value : float, optional
        Use a fill value for points outside of input volume. Default=0.

    force_resample : bool, optional
        Intended for testing, this prevents the use of a padding optimzation.
        Default=False.

    Returns
    -------
    resampled: nibabel.Nifti1Image
        input image, resampled to have respectively target image shape and
        affine as shape and affine.

    See Also
    --------
    nilearn.image.resample_img

    """
    target = _utils.check_niimg(target_img)
    target_shape = target.shape

    # When target shape is greater than 3, we reduce to 3, to be compatible
    # with underlying call to resample_img
    if len(target_shape) > 3:
        target_shape = target.shape[:3]

    return resample_img(source_img,
                        target_affine=target.affine,
                        target_shape=target_shape,
                        interpolation=interpolation, copy=copy, order=order,
                        clip=clip, fill_value=fill_value,
                        force_resample=force_resample)
