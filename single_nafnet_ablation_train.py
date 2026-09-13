import os

from reverse_distill_unet import main


if __name__ == '__main__':
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main('./options/train/singleNAFNetDirectXT.yml')