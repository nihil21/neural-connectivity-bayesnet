# first line: 640
def compute_multi_gray_matter_mask(target_imgs, threshold=.5,
                                   connected=True, opening=2,
                                   memory=None, verbose=0, n_jobs=1, **kwargs):
    """ Compute a mask corresponding to the gray matter part of the brain for
    a list of images.
    The gray matter part is calculated through the resampling of MNI152
    template gray matter mask onto the target image

    Parameters
    ----------
    target_imgs: list of Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Images used to compute the mask. 3D and 4D images are accepted.
        The images in this list must be of same shape and affine. The mask is
        calculated with the first element of the list for only the shape/affine
        of the image is used for this masking strategy

    threshold: float, optional
        The value under which the MNI template is cut off.
        Default value is 0.5

    connected: bool, optional
        if connected is True, only the largest connect component is kept.
        Default is True

    opening: bool or int, optional
        if opening is True, a morphological opening is performed, to keep
        only large structures.
        If opening is an integer `n`, it is performed via `n` erosions.
        After estimation of the largest connected constituent, 2`n` closing
        operations are performed followed by `n` erosions. This corresponds
        to 1 opening operation of order `n` followed by a closing operator
        of order `n`.

    memory: instance of joblib.Memory or str
        Used to cache the function call.

    n_jobs: integer, optional
        Argument not used but kept to fit the API

    **kwargs: optional arguments
        arguments such as 'target_affine' are used in the call of other
        masking strategies, which then would raise an error for this function
        which does not need such arguments.

    verbose: int, optional
        Controls the amount of verbosity: higher numbers give
        more messages

    Returns
    -------
    mask: nibabel.Nifti1Image
        The brain mask (3D image)

    See also
    --------
    nilearn.masking.compute_brain_mask
    """
    if len(target_imgs) == 0:
        raise TypeError('An empty object - %r - was passed instead of an '
                        'image or a list of images' % target_imgs)

    # Check images in the list have the same FOV without loading them in memory
    imgs_generator = _utils.check_niimg(target_imgs, return_iterator=True)
    for _ in imgs_generator:
        pass

    mask = compute_brain_mask(target_imgs[0], threshold=threshold,
                                    connected=connected, opening=opening,
                                    memory=memory, verbose=verbose)
    return mask
