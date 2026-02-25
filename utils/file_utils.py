def none_or_true(value):
    if value == 'None':
        return None
    elif value == "True":
        return True
    return value
