import logging

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器并设置级别为DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建格式化器并将其添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger