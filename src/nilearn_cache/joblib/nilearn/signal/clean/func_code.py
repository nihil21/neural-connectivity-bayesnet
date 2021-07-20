# first line: 32
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _warn_deprecated_params(replacement_params, end_version, lib_name,
                                    kwargs
                                    )
            kwargs = _transfer_deprecated_param_vals(replacement_params,
                                                     kwargs
                                                     )
            return func(*args, **kwargs)
