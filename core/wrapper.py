def benchmark(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end - start))

    return wrapper


@benchmark
def fetch_webpage(a, b, c):
    import requests
    webpage = requests.get('https://google.com')


fetch_webpage(1, 2, 3)
