# coding=utf-8
import os


def init_distributed_mode(args):
    """
    Handle single and multi-GPU
    """
    # multi-GPU
    if args.rank != -1:
        print("local rank:%s" % args.local_rank)
        args.distributed = True
        # read environment variables
        args.world_size = os.environ.get("WORLD_SIZE", args.world_size)
        args.master_addr = os.environ.get("MASTER_ADDR", args.master_addr)
        args.master_port = os.environ.get("MASTER_PORT", args.master_port)

        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ['WORLD_SIZE'] = args.WORLD_SIZE
        os.environ['MASTER_PORT'] = args.master_port

        # number of nodes / node ID
        args.n_gpu_per_node = args.world_size // args.nodes
    else:
        assert args.local_rank == -1

        args.distributed = False
        args.n_nodes = 1
        args.node_id = 0
        args.local_rank = 0
        args.world_size = 1
        args.n_gpu_per_node = 1

    assert args.n_nodes >= 1
    assert 0 <= args.node_id < args.n_nodes
