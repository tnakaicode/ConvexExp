import numpy as np
import time
import os
import sys
import tqdm


class ProgressBarPrinter:
    def __init__(self, width, step, stream, fname):
        self.width = width
        self.block_progress = 0
        self.current_progress = 0
        self.start_time = time.time()
        self.step = step
        self.stream = stream

        # Prints the progress bar's layout
        print(f"{fname}: [{'-' * self.width}] 0%% <0.00 seconds>",
              flush=True, end='', file=self.stream)
        print("\b" * (self.width + len("] 0%% <0.00 seconds>")),
              end='', file=self.stream)

    def update(self, progress):
        # Parsing input
        if progress is None:
            progress = self.current_progress + self.step
        if not isinstance(progress, float):
            raise TypeError("ProgressBar: input must be float or None")

        # Keep the progress bar under 99% until end() has been called
        self.current_progress = min(progress, 0.99)
        self.print_bar(self.current_progress)

    def print_bar(self, progress):
        block = int(round(self.width * progress)) - self.block_progress
        self.block_progress += block
        bar = ('#' * block) + ('-' * (self.width - self.block_progress))
        progress = int(progress * 100)
        elapsed_time = round(time.time() - self.start_time, 2)
        text = f"{bar}] {progress}% <{elapsed_time} seconds>"
        print(text + ("\b" * (len(text) - block)),
              flush=True, end='', file=self.stream)

    def end(self):
        self.print_bar(1.0)
        print(flush=True, file=self.stream)


def ProgressBar(width=50, step=0.1, stream=sys.stdout):
    """Decorator, prints a progress bar when a decored function yields it's
    current progress.
    When you want the progress bar to be updated you should yield the progress
    of your function between 0 and 1. The general calcul for this is:
    (current_iteration + 1) / total_iterations.
    When yielding None, the progress bar goes up by `current progress + step`.
    This is usefull to show some feedback instead of a dead terminal when it's
    not possible to calculate the progress of a function.
    Limitation: It uses yield statements as callbacks for the decorator. That
    means you can't yield your result, meaning this progress bar doesn't
    work if your function is intended to be a generator.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            pb = ProgressBarPrinter(width, step, stream, func.__name__)
            progress_generator = func(*args, **kwargs)
            try:
                while True:
                    progress = next(progress_generator)
                    pb.update(progress)
            except StopIteration as result:
                pb.end()
                return result.value
        return wrapper
    return decorator


if __name__ == "__main__":
    for i in tqdm.tqdm(range(10)):
        for j in tqdm.tqdm(range(10), leave=False):
            np.pi * np.pi
            time.sleep(1 / 2**4)

    @ProgressBar()  # Decorate the function with the progress bar
    def dummyLoop():
        nb_iter = 13
        for i in range(nb_iter):
            time.sleep(0.1)
            yield ((i + 1) / nb_iter)  # Give control back to decorator
        # returning the real result
        return "My result"

    res = dummyLoop()  # You can still retrieve the result of your function
    print("result:", res)

    @ProgressBar(step=0.1, stream=sys.stderr)
    def yieldNaN():
        nb_iter = 11
        for i in range(nb_iter):
            time.sleep(0.1)
            yield  # Give control back to decorator and increase by step
        return None

    print("result:", yieldNaN())
