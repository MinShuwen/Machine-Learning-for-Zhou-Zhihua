'''
带后剪枝的Cart决策树
在生成一颗不剪枝的决策树之后，对每一个满足其所有子节点都为叶子节点的节点进行判断，
计算如果将其子节点全部删除能否带来决策树对测试数据分类正确率的提高，如果能，则进行剪枝
一般来说，后剪枝的效果是比预剪枝的好的
'''
import TreeNode
import Cart
import DataSet

def current_acc(root_node, test_data, test_label):
    '''
    计算当前决策树在训练数据集上的正确率
    :param rootNode: 决策树的根节点
    :param test_data: 测试数据集
    :param test_label: 测试数据集的label
    :return:
    '''
    # 每次都从根节点开始判断，则直接将tree_node设置为decesion_tree
    # root_node = tree_node
    # while not(root_node.parent is None):
    #     root_node = root_node.parent

    acc = 0
    for i in range(len(test_label)):
        this_label = Cart.classify_data(root_node, test_data[i])
        if this_label == test_label[i]:
            acc += 1
    return acc/len(test_label)

def post_pruning(decesion_tree, test_data, test_label, train_label):
    '''
    对决策树进行后剪枝操作，对已经构建好的决策树进行判断是否进行属性划分
    :param decesion_tree: 决策树根节点
    :param test_data: 测试数据集
    :param test_label: 测试数据集的标签
    :param train_label: 训练数据集的标签
    :return:
    '''
    leaf_father = []  # 所有孩子都是叶节点的节点集合

    help_list = []
    help_list.append(decesion_tree)
    while len(help_list)>0:
        cur_node = help_list.pop(0)
        children = cur_node.children
        flag = True  # 标记当前节点是否满足所有的子节点都是叶子节点
        if not (children is None):
            for child in children:
                help_list.append(child)
                flag1 = (child.children is None)  # 孩子的孩子是否为空
                flag = (flag and flag1)
        else:
            flag = False

        if flag:
            leaf_father.append(cur_node)

    while len(leaf_father)>0:
        # 如果父节点为空，则剪枝完成。对于不需要进行剪枝操作的节点，也需要从leaf_father列表中删除
        cur_node = leaf_father.pop()
        # 进行剪枝在测试集上的正确率
        before_acc = current_acc(root_node=decesion_tree, test_data=test_data, test_label=test_label)
        # 如果不剪枝
        data_index = cur_node.data_index
        label_count = {}
        for index in data_index:
            if label_count.__contains__(train_label[index]):
                label_count[train_label[index]] += 1
            else:
                label_count[train_label[index]] = 1
        cur_node.judge = max(label_count, key=label_count.get)
        later_acc = current_acc(decesion_tree, test_data, test_label)

        # 比较剪枝前后的准确率
        if before_acc > later_acc:
            cur_node.judge = None
        else:
            cur_node.children = None
            # 还需要检查事都需要对他的父节点进行判断
            parent_node = cur_node.parent
            if not (parent_node is None):
                children_list = parent_node.children
                tmp_flag = True
                for child in children_list:
                    if not(child.children is None):
                        tmp_flag = False
                        break
                if tmp_flag:
                    leaf_father.append(parent_node)
    return decesion_tree


def run_test():
    train, test, title = DataSet.watermelon2()

    train_data = []
    test_data = []
    train_lable = []
    test_label = []
    for t in train:
        a_dict = {}
        dim = len(t)-1 # 训练数据大小
        for i in range(dim):
            a_dict[title[i]] = t[i]
        train_data.append(a_dict)
        train_lable.append(t[dim])
    for t in test:
        a_dict = {}
        dim = len(t)-1
        for i in range(dim):
            a_dict[title[i]] = t[i]
        test_data.append(a_dict)
        test_label.append(t[dim])

    decision_tree = Cart.cart_tree(train_data, title, train_lable)
    decision_tree = post_pruning(decision_tree, test_data, test_label, train_lable)
    print('后剪枝之后的决策树是：')
    Cart.print_tree(decision_tree)
    print('\n')

    test_judge = []
    for melon in test_data:
        test_judge.append(Cart.classify_data(decision_tree,melon))
    print('决策树在测试数据集上的分类结果是：',test_judge)
    print('测试数据集的正确类别信息是：',test_label)

    acc = 0
    for i in range(len(test_label)):
        if test_label[i] == test_judge[i]:
            acc += 1
    acc /= len(test_label)
    print('决策树在测试数据集上的分类正确率为：',str(acc*100)+'%')


if __name__ == '__main__':
    '''
    run_test()->cart_tree()->post_pruning()->current_acc()
              ->print_tree()
    '''
    run_test()