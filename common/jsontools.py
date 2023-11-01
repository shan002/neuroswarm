import json


def smartload(x) -> dict:
    """
    Parse a JSON string or file and return a dict.

    Interpret the input as either a JSON str or a JSON file in a
    robust manner, and give useful errors otherwise.

    args:
    x should be a string containing valid JSON, or a file descriptor (path string).

    NB: Interpreting x as a file takes precedence over interpreting x
        as a JSON str. If x is '{"a": null}' and this is the name of an
        actual file containing valid JSON, the latter will be parsed.
    """
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return json.loads(x)  # try this i guess

    # x is now known to be a string.
    file_error = None
    parse_error = None
    file_parse_error = None

    try:
        f = open(x, 'r')
    except BaseException as e:
        file_error = e
    else:
        # x is a valid file and can be opened
        file_error = False
        try:
            return json.loads(f.read())
        except BaseException as e:
            file_parse_error = e
    finally:
        f.close()

    # if we're here, either file read or file JSON parsing failed.

    try:
        return json.loads(x)
    except BaseException as e:
        parse_error = e

    # if we're here, parsing x:str as JSON failed and x is not a valid JSON file.
    errors = filter([parse_error, file_error, file_parse_error])
    try:
        ExceptionGroup
    except NameError:  # this python doesn't have ExceptionGroup
        raise Exception(errors)
    else:
        raise ExceptionGroup(errors)  # new in python 3.11