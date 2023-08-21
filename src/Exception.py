import sys


def get_exception(error, error_msg: sys):
    _, _, exc_tb = error_msg.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"\nCustom Exception raised in {filename} at line number: {line_no},\nerror: {error}"


class CustomException(Exception):

    def __init__(self, error, error_msg: sys):
        super().__init__(error)
        self.error = get_exception(error, error_msg)

    def __str__(self):
        return self.error
