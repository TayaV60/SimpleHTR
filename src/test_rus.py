import datetime
from path import Path

data_dir=Path("/Users/taisiyavelarde/Documents/rus-iam-format")

def main():
    f = open(data_dir / 'gt/words.txt')
    for line in f:
        if not line or line[0] == '#':
            continue
        line_split = line.strip().split(' ')
        print (line)
        assert len(line_split) >= 9

if __name__ == '__main__':
    print("STARTED")
    print(datetime.datetime.now())
    main()
    print("ENDED")
    print(datetime.datetime.now())
