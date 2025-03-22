# -*- coding: utf-8 -*-
'''
@File    : redirect_std_to_file.py
@Time    : 2025/02/27 14:09:38
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : For each python process,
use `redirect_std_to_file(path_log)` once time,
redirect stdout and stderr into path_log file.
'''
import sys
class Logger:
  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = open(filename, 'a', encoding='utf-8')

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def redirect_std_to_file(filename):
  sys.stderr = sys.stdout = Logger(filename)

if __name__ == '__main__':
  redirect_std_to_file('test.log')
  print("hi")
  raise ValueError("GG")