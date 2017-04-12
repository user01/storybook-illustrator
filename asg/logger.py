# -*- coding: utf-8 -*-
"""Utility functions for logging"""

import datetime
import time


TIME_LAST = None

class Logger:
    _time_last = None

    @staticmethod
    def _human_seconds(seconds):
        """Convert a number of seconds to a human readable string"""
        if seconds < 60:
            return '{: >5.1f} seconds'.format(seconds)
        if seconds < 60*60:
            return '{: >5.1f} minutes'.format(seconds / 60)
        return '{: >7.1f} hours'.format(seconds / (60 * 60))


    @staticmethod
    def log(content):
        """Log a value with the current time"""

        now = datetime.datetime.now().strftime("%c")
        now_time = time.time()
        # msg_last = '{} - {: >5.1f} seconds - {}'.format(now, now_time - TIME_LAST, content)

        if Logger._time_last is not None:
            msg_last = Logger._human_seconds(now_time - Logger._time_last)
        else:
            msg_last = ' ' * 13

        msgs = [now, msg_last, content]

        msg = " │ ".join(msgs)

        msg_lines = ["─" * len(content) for content in msgs]

        msg_top = "─┬─".join(msg_lines)
        msg_lower = "─┴─".join(msg_lines)

        print(" ┌─{}─┐".format(msg_top))
        print(" │ {} │".format(msg))
        print(" └─{}─┘".format(msg_lower))

        Logger._time_last = time.time()