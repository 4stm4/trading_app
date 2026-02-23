from datetime import datetime
curr_dt = datetime.now()
before_dt = datetime.strptime('14-03-24 20:43:22', '%d-%m-%y %H:%M:%S')
before_dt = datetime.strptime('14-03-2024 20:43:22', '%d-%m-%Y %H:%M:%S')

timestamp = int(round(curr_dt.timestamp()))
print("Integer timestamp of current datetime: ",timestamp)
print('before: ', int(round(before_dt.timestamp())))

def dt_string_to_timestamp(dt_str: str) -> int:
    dt = None
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
    except Exception as error:
        print(error)    # FIXME: сменить на логирование
    return int(round(dt.timestamp())) if dt else 0

print(datetime.now())
print(dt_string_to_timestamp(str(datetime.now())))