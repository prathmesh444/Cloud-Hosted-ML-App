import sys


def get_exception(error, error_msg: sys):
    filename = error_msg.exc_info()[-1].tb_frame.f_code.co_filename,
    line_no = error_msg.exc_info()[-1].tb_lineno,
    return f"Custom Exception raised in {filename} at line number: {line_no}, error: {str(error)}"


class CustomException(Exception):
    def __int__(self, error, error_msg: sys):
        self.error = error
        self.error_msg = get_exception(error, error_msg)

    def __str__(self):
        return self.error_msg
