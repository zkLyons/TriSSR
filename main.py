import sys
import torch
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.config import Config
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

from custom_utils import TriSSRData_preparation, TriSSRDataset
from custom_trainer import TriSSRTrainer
from trissr import TriSSR



if __name__ == '__main__':
    config = Config(model=TriSSR, config_file_list=['./config.yaml'])
    init_seed(config['seed'], config['reproducibility']) # 2024, True
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering, 创建原始数据集，清晰数据等
    dataset = TriSSRDataset(config)
    logger.info(dataset)

    # dataset splitting  切分数据集，并创建对应的dataloader
    train_data, valid_data, test_data = TriSSRData_preparation(config, dataset)

    # model loading and initialization 加载模型和初始化
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = TriSSR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = TriSSRTrainer(config, model)


    # 在模型初始化后，训练开始前添加
    torch.backends.cuda.cufft_plan_cache.clear()
    print("cuFFT缓存已清除")

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"] # config["show_progress"]: True
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    # 整体执行的逻辑为：训练1轮（优化参数）-》测试（锁定参数）-》输出测评结果-》训练2轮（优化参数）...->最后使用测试集进行一轮测试，的到最后的结果，输出。