import os.path
from copy import deepcopy
import torch
from config import get_train_args
from utils.loss import all_snapshots_loss
from utils.metrics import is_better, null_metrics, compute_metrics_one_snapshot
from models.model import BotSTIP
from utils.dataset import Dataset
from pytorch_lightning import seed_everything
from tqdm import tqdm


class Trainer:
    def __init__(self, args):
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean') # 创建一个损失函数 CrossEntropyLoss，用于计算交叉熵损失，reduction='mean' 表示计算平均损失
        self.dataset = Dataset(self.args.dataset_name, self.args.interval, self.args.batch_size, self.args.seed,
                               self.args.window_size, self.args.device) # 初始化一个数据集对象 Dataset，传入参数如数据集名称、间隔、批大小等
        self.des_tensor, self.tweets_tensor, self.num_prop, self.category_prop, self.labels = self.dataset.des_tensor, self.dataset.tweets_tensor, self.dataset.num_prop, self.dataset.category_prop, self.dataset.labels
        self.train_right, self.train_n_id, self.train_edge_index, self.train_edge_type, self.train_exist_nodes, self.train_clustering_coefficient, self.train_bidirectional_links_ratio = self.dataset.train_right, self.dataset.train_n_id, self.dataset.train_edge_index, self.dataset.train_edge_type, self.dataset.train_exist_nodes, self.dataset.train_clustering_coefficient, self.dataset.train_bidirectional_links_ratio
        self.test_right, self.test_n_id, self.test_edge_index, self.test_edge_type, self.test_exist_nodes, self.test_clustering_coefficient, self.test_bidirectional_links_ratio = self.dataset.test_right, self.dataset.test_n_id, self.dataset.test_edge_index, self.dataset.test_edge_type, self.dataset.test_exist_nodes, self.dataset.test_clustering_coefficient, self.dataset.test_bidirectional_links_ratio
        self.val_right, self.val_n_id, self.val_edge_index, self.val_edge_type, self.val_exist_nodes, self.val_clustering_coefficient, self.val_bidirectional_links_ratio = self.dataset.val_right, self.dataset.val_n_id, self.dataset.val_edge_index, self.dataset.val_edge_type, self.dataset.val_exist_nodes, self.dataset.val_clustering_coefficient, self.dataset.val_bidirectional_links_ratio
        if self.args.dataset_name == 'Twibot-20':
            self.labels = torch.cat(
                (self.labels, 3 * torch.ones(229580 - len(self.labels), device=self.args.device).long()), dim=0) # 如果数据集名称为 Twibot-20，则将标签张量扩展到 229580 个元素，使用 3 填充剩余部分。
        self.args.window_size = self.dataset.window_size
        self.model = BotSTIP(self.args)  # 实例化模型 BotDyGNN，并传入之前的参数。
        self.model.to(self.args.device)  # 移动到GPU
        params = [
            {"params": self.model.node_feature_embedding_layer.parameters(), "lr": self.args.structural_learning_rate}, 
            {"params": self.model.structural_layer.parameters(), "lr": self.args.structural_learning_rate},
            {"params": self.model.temporal_layer.parameters(), "lr": self.args.temporal_learning_rate},
        ]  # 定义优化器的参数，包括每个层的学习率设置。
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0) # 定义学习率调度器 CosineAnnealingLR，用于调整学习率。
        self.pbar = range(self.args.epoch) # 初始化一个进度条范围，表示训练的轮数。
        self.best_val_metrics = null_metrics() # 初始化最佳验证指标，使用 null_metrics 函数（假设这个函数返回一个空的度量结构）。
        self.test_state_dict_list = [] # 初始化一个空列表，用于保存测试模型的状态字典。
        self.test_epoch = None
        self.test_epoch_list = []
        self.test_metrics = null_metrics()
        self.test_state_dict = None # 初始化测试的状态字典，起始为 None。
        self.last_state_dict = None # 初始化最后的状态字典，起始为 None。

    def forward_one_batch(self, batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                          batch_clustering_coefficient, batch_bidirectional_links_ratio):  # 这行定义了一个名为 forward_one_batch 的方法，接受多个参数，包括批次大小 batch_size、节点ID batch_n_id、边的信息 batch_edge_index、存在节点的信息 batch_exist_nodes、聚类系数 batch_clustering_coefficient 和双向链接比率 batch_bidirectional_links_ratio
        des_tensor_list = [self.des_tensor[n_id].to(self.args.device) for n_id in batch_n_id] # 创建一个列表 des_tensor_list，在这个列表中，将 self.des_tensor 中对应 batch_n_id 的每个节点ID的张量移动到指定设备（如GPU）。
        tweet_tensor_list = [self.tweets_tensor[n_id].to(self.args.device) for n_id in batch_n_id] # 建 tweet_tensor_list，类似地，把 self.tweets_tensor 中对应的每个节点张量移动到设备
        num_prop_list = [self.num_prop[n_id].to(self.args.device) for n_id in batch_n_id] # 创建 num_prop_list，将 self.num_prop 中对应的节点属性张量移动到设备。
        category_prop_list = [self.category_prop[n_id].to(self.args.device) for n_id in batch_n_id] # 创建 category_prop_list，将 self.category_prop 中对应的节点类别属性张量移动到设备
        label_list = [self.labels[n_id][:batch_size].to(self.args.device) for n_id in batch_n_id] # 创建 label_list，从 self.labels 中获取对应节点ID的标签，并且只取前 batch_size 个标签，移动到设备。
        label_list = torch.stack(label_list, dim=0) # 使用 torch.stack 函数将 label_list 中的所有张量沿第0维合并成一个新的张量。
        edge_index_list = [_.to(self.args.device) for _ in batch_edge_index] # 将 batch_edge_index 中的每个边张量移动到设备，并创建一个新的列表 edge_index_list。
        clustering_coefficient_list = [_.to(self.args.device) for _ in batch_clustering_coefficient] # 将 batch_clustering_coefficient 中的每个聚类系数张量移动到设备，并创建相应的列表 clustering_coefficient_list。
        bidirectional_links_ratio_list = [_.to(self.args.device) for _ in batch_bidirectional_links_ratio] # 将 batch_bidirectional_links_ratio 中的双向链接比率张量移动到设备，并创建列表 bidirectional_links_ratio_list。
        exist_nodes_list = [exist_nodes[:batch_size].to(self.args.device) for exist_nodes in batch_exist_nodes] # 从 batch_exist_nodes 中获取每个存在节点的信息，取前 batch_size 个，并将其移动到设备，形成新列表 exist_nodes_list
        exist_nodes_list = torch.stack(exist_nodes_list, dim=0) 
        output = self.model(des_tensor_list, tweet_tensor_list, num_prop_list, category_prop_list, edge_index_list,
                               clustering_coefficient_list, bidirectional_links_ratio_list, exist_nodes_list,
                               batch_size)  # [64,13,2]
        output = output.transpose(0, 1) # [13,64,2]
        loss = all_snapshots_loss(self.criterion, output, label_list, exist_nodes_list)
        return output, loss, label_list, exist_nodes_list

    def forward_one_epoch(self, right, n_id, edge_index, exist_nodes, clustering_coefficient,
                          bidirectional_links_ratio):
        all_label = []
        all_output = []
        all_exist_nodes = []    # 初始化三个空列表 all_label、all_output 和 all_exist_nodes 用于存储每个批次的标签、输出和存在节点的信息。同时，初始化 total_loss 变量用于累加总损失。
        total_loss = 0.0
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, batch_clustering_coefficient, batch_bidirectional_links_ratio \
                in zip(right, n_id, edge_index, exist_nodes, clustering_coefficient, bidirectional_links_ratio):   
            output, loss, label_list, exist_nodes_list = self.forward_one_batch(batch_size, batch_n_id,
                                                                                batch_edge_index, batch_exist_nodes,
                                                                                batch_clustering_coefficient,
                                                                                batch_bidirectional_links_ratio)
            total_loss += loss.item() / self.args.window_size / len(right)
            all_output.append(output)
            all_label.append(label_list)
            all_exist_nodes.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1) 
        print("all_output1.shape",all_output.shape)
        all_label = torch.cat(all_label, dim=1)
        all_exist_nodes = torch.cat(all_exist_nodes, dim=1)   # 将当前批次的输出、标签和存在节点分别添加到相应的总列表中
       
        metrics = compute_metrics_one_snapshot(all_label[-1], all_output[-1], exist_nodes=all_exist_nodes[-1]) # 调用 compute_metrics_one_snapshot 函数，计算最后一批的输出与标签的指标（如准确率、损失等），并将存在节点作为参数传入。
        metrics['loss'] = total_loss
        return metrics

    def train_per_epoch(self, current_epoch):
        self.model.train()  # 将模型设置为训练模式。这样，某些层（如 Dropout 和 BatchNorm）将在训练时以特定方式工作。
        all_label = []
        all_output = []
        all_exist_nodes = []
        total_loss = 0.0
        plog = ""    # 初始化几个空列表来保存每个批次的标签、输出和存在的节点信息，并且初始化总损失为 0。plog 用于保存日志信息。
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, batch_clustering_coefficient, batch_bidirectional_links_ratio \
                in tqdm(zip(self.train_right, self.train_n_id, self.train_edge_index, self.train_exist_nodes,
                       self.train_clustering_coefficient, self.train_bidirectional_links_ratio)):  # 使用 zip 将多个训练数据的属性（例如：batch_size、batch_n_id 等）组合成一个迭代器，每次循环都取出一个批次的数据。
            self.optimizer.zero_grad()  # 在每个批次开始前，将优化器的梯度清零，以避免上一个批次的梯度对当前批次的影响。
            output, loss, label_list, exist_nodes_list = self.forward_one_batch(batch_size, batch_n_id,
                                                                                batch_edge_index, batch_exist_nodes,
                                                                                batch_clustering_coefficient,
                                                                                batch_bidirectional_links_ratio)  # 调用 forward_one_batch 方法来计算当前批次的输出、损失、标签列表和存在的节点列表。
            total_loss += loss.item() / self.args.window_size / len(self.train_right)  # 将当前批次的损失加到总损失中。这里还将损失除以 window_size 和训练数据的长度，以归一化损失。
            loss.backward() # 反向传播以计算梯度。
            self.optimizer.step()  # 更新模型参数
            all_output.append(output)
            all_label.append(label_list)
            all_exist_nodes.append(exist_nodes_list)  # 将当前批次的输出、标签和存在的节点添加到对应的列表中，以便后续使用。
        all_output = torch.cat(all_output, dim=1)
        print("all_output2.shape",all_output.shape)
        torch.save(all_output, "all_output_BotSTIP.pt")
        all_label = torch.cat(all_label, dim=1)
        torch.save(all_label, "all_label_BotSTIP.pt")
        # print("all_label:", all_label.shape)
        all_exist_nodes = torch.cat(all_exist_nodes, dim=1)    # 将所有批次的输出、标签和存在的节点按列拼接在一起。这使得我们可以在整个训练过程中计算综合指标。
        # print("all_exist_nodes:", all_exist_nodes.shape)
        metrics = compute_metrics_one_snapshot(all_label[-1], all_output[-1], exist_nodes=all_exist_nodes[-1])  # 计算模型在当前快照下的性能指标，使用最后一个批次的标签和输出。
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            plog += ' {}: {:.6}'.format(key, metrics[key])   # 将计算得到的指标（准确率、精确率、召回率和 F1 值）格式化并添加到日志字符串中。
        plog = 'Epoch-{} train loss: {:.6}'.format(current_epoch, total_loss) + plog  # 将当前轮次的训练损失添加到日志中
        print(plog)
        metrics['loss'] = total_loss  # 将总损失添加到性能指标字典中。
        return metrics

    @torch.no_grad()
    def val_per_epoch(self, current_epoch): # 定义一个方法 val_per_epoch，用来在每个训练周期（epoch）后进行验证。current_epoch 参数表示当前的训练周期数。
        self.model.eval()  # 将模型设置为评估模式。这意味着在这个模式下，某些层（如Dropout和BatchNorm）会表现得不一样，以适应验证或测试阶段。
        metrics = self.forward_one_epoch(self.val_right, self.val_n_id, self.val_edge_index, self.val_exist_nodes,
                                         self.val_clustering_coefficient, self.val_bidirectional_links_ratio) # 调用 forward_one_epoch 方法来进行一轮前向传播，使用验证集的数据。这会返回一个包含各种性能指标（如损失、准确率等）的字典 metrics。
        plog = "" # 初始化一个空字符串 plog，用于后面日志信息的构建。
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            plog += ' {}: {:.6}'.format(key, metrics[key])  # 开始一个循环，遍历一个列表，该列表包含四个性能指标的名称：准确率（accuracy）、精确度（precision）、召回率（recall）和F1分数（f1）
        plog = 'Epoch-{} val loss: {:.6}'.format(current_epoch, metrics['loss']) + plog  # 将当前指标的名称和对应的值格式化为字符串，并添加到 plog 中。{:.6} 表示保留六位小数。
        print(plog)
        return metrics

    @torch.no_grad()  # 这是一个装饰器，用于指示PyTorch在执行这个函数时不计算梯度。这可以节省内存和计算资源，因为在验证和测试阶段我们不需要更新模型的权重。
    def test_last_model(self):
        self.model.load_state_dict(self.last_state_dict)
        self.model.eval()
        metrics = self.forward_one_epoch(self.test_right, self.test_n_id, self.test_edge_index, self.test_exist_nodes,
                                         self.test_clustering_coefficient, self.test_bidirectional_links_ratio)
        plog = ""
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            plog += ' {}: {:.6}'.format(key, metrics[key])
        plog = 'Last Epoch test loss: {:.6}'.format(metrics['loss']) + plog
        print(plog)
        return metrics, self.last_state_dict

    @torch.no_grad()
    def test_best_model(self, top_k=1):
        best_test_metrics = null_metrics()
        best_test_state_dict = None
        self.test_state_dict_list = self.test_state_dict_list[-top_k:]
        self.test_epoch_list = self.test_epoch_list[-top_k:]
        print('start testing...')
        for epoch, state_dict in zip(self.test_epoch_list, self.test_state_dict_list):
            self.model.load_state_dict(state_dict)
            self.model.eval()
            metrics = self.forward_one_epoch(self.test_right, self.test_n_id, self.test_edge_index,
                                             self.test_exist_nodes, self.test_clustering_coefficient,
                                             self.test_bidirectional_links_ratio)
            plog = ""
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                plog += ' {}: {:.6}'.format(key, metrics[key])
            plog = 'Epoch-{} test loss: {:.6}'.format(epoch, metrics['loss']) + plog
            print(plog)
            if is_better(metrics, best_test_metrics):
                best_test_metrics = metrics
                best_test_state_dict = state_dict
        return best_test_metrics, best_test_state_dict

    def train(self):
        validate_score_non_improvement_count = 0 # 初始化一个计数器，用于跟踪验证集评分没有改进的次数
        self.model.train() # 将模型设置为训练模式。这在 PyTorch 中是必要的，因为某些层（如 dropout 和 batch normalization）在训练和测试模式下的行为是不同的。
        for current_epoch in (self.pbar): # 使用一个循环来遍历每个训练的周期（epoch）。self.pbar 应该是一个进度条
            self.train_per_epoch(current_epoch) # 调用train_per_epoch方法，传入当前的周期。这个方法负责在这一周期内训练模型。
            self.scheduler.step() # 调用学习率调度器的step方法，通常这是在每个周期结束后调整学习率。
            val_metrics = self.val_per_epoch(current_epoch) # 调用val_per_epoch方法，传入当前的周期，获取验证指标（如准确率、损失等）
            if is_better(val_metrics, self.best_val_metrics): # 使用is_better函数来比较当前验证指标与之前最好的验证指标。如果当前更好，就进入条件内部。
                self.best_val_metrics = val_metrics  
                self.test_epoch = current_epoch  # 记录当前周期为测试周期。
                self.test_epoch_list.append(current_epoch) # 将当前周期添加到测试周期列表中。
                self.test_state_dict = deepcopy(self.model.state_dict()) # 深拷贝模型的状态字典（即模型的参数）并保存到test_state_dict中。
                self.test_state_dict_list.append(self.test_state_dict) # 将当前的状态字典添加到状态字典列表中，以便后续使用
                validate_score_non_improvement_count = 0  # 如果有改进，则重置计数器为0。
            else:
                validate_score_non_improvement_count += 1  # 如果没有改进，则计数器加1。
            self.last_state_dict = deepcopy(self.model.state_dict()) # 在每个周期结束时，保存当前模型的状态字典，以备后续使用。
            if self.args.early_stop and validate_score_non_improvement_count >= self.args.patience: # 检查是否启用提前停止机制，并判断是否超过了耐心值（即阈值）来决定是否停止训练。
                print('Early stopping at epoch: {}'.format(current_epoch))  # 如果满足提前停止的条件，则打印出提前停止的周期。
                break
        if self.args.early_stop:  # 检查是否启用了提前停止机制。
            self.test_metrics, self.test_state_dict = self.test_best_model(top_k=1) # 如果启用，测试最好的模型并获取其指标和状态字典。
        else:
            self.test_metrics, self.test_state_dict = self.test_last_model() # 如果没有启用提前停止，测试最近的模型并获取其指标和状态字典
        model_name = f"{self.args.interval} + {self.args.seed} + {self.test_metrics['accuracy']} + {self.test_metrics['precision']} + {self.test_metrics['f1']}.pt "  # 根据一些参数和性能指标生成模型名称。
        model_dir_path = os.path.join('output', self.args.dataset_name) # 构建保存模型的目录路径。
        if not os.path.exists(model_dir_path): # 如果目录不存在，则创建该目录。
            os.makedirs(model_dir_path)
        torch.save(self.test_state_dict, os.path.join(model_dir_path, model_name)) # 将模型的状态字典保存到指定路径下。


def main(args):
    seed_everything(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_train_args()
    print(args)
    main(args)
