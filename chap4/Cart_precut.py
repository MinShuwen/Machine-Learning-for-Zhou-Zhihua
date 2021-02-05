'''
带预剪枝的Cart决策树
预剪枝指在决策树的生成过程中，对每个节点在划分前后先进性评估，
如果当前节点的划分不能带来决策树泛化性能的提升，则停止划分，并将当前节点标记为叶节点。
也就是对每个节点比较这个节点进行划分和不进行划分两种情况下决策树在测试数据集上的正确率。
'''

import TreeNode
import DataSet
import Cart

def current_accuracy(tree_node = None, test_data = None, test_label = None):
    '''
    计算当前决策树在训练数据集上的正确率
    :param tree_node: 要判断的决策树节点
    :param test_data: 测试数据集
    :param test_label: 测试数据集的label
    :return:
    '''
    root_node = tree_node
    while not root_node.parent is None:
        root_node = root_node.parent
    acc = 0
    for i in range(len(test_label)):
        this_label = Cart.classify_data(root_node,test_data[i])
        if this_label == test_label[i]:
            acc += 1
    return acc/len(test_label)

def finish_node(current_node = None, data = None, label = None, test_data = None, test_label = None):
    '''
    完成一个节点上的计算，预剪枝需要在构建决策树时判断是否进行属性划分
    :param current_node:当前计算的节点
    :param data: 数据集
    :param label: 数据集上的label
    :param test_data: 测试数据集
    :param test_label: 测试数据集的label
    :return:
    '''
    n = len(label)

    # 判断当前节点中的数据是否属于同一类，如果为同一类，则直接标记为叶子节点
    one_class = True
    this_data_index = current_node.data_index

    for i in this_data_index:
        for j in this_data_index:
            if label[i]!=label[j]:
                one_class = False
                break
        if not one_class:
            break
    if one_class:
        current_node.judge = label[this_data_index[0]]
        return

    rest_title = current_node.rest_attr  # 候选属性
    if len(rest_title) == 0:
        # 如果候选属性为空，则标记为叶子节点，选择最多的那个类别作为该节点的类
        label_count = {} # {label:count}
        tmp_data = current_node.data_index
        for index in tmp_data:
            if label_count.__contains__(label[index]):
                label_count[label[index]] += 1
            else:
                label_count[label[index]] = 1
        final_label = max(label_count, key=label_count.get)
        # max(dict,key=dict.get) 则求出来的最大值为 最大的value对应的key
        current_node.judge = final_label
        return

    # 先为当前节点添加一个临时判断，如果需要添加孩子节点，就把他恢复为None
    data_count = {}
    for index in current_node.data_index:
        if data_count.__contains__(label[index]):
            data_count[label[index]] += 1
        else:
            data_count[label[index]] = 1
    before_judge = max(data_count,key=data_count.get)
    current_node.judge = before_judge
    before_acc = current_accuracy(current_node,test_data,test_label)  # 不剪枝的正确率

    title_gini = {} # 记录每个属性的gini指数
    title_split_value = {} # 记录每个属性的分割值，如果是连续属性，则为分割值，如果是离散属性，则为None
    for title in rest_title:
        attr_values = []
        cur_labels = []
        for index in current_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            cur_labels.append(label[index])
        tmp_data = data[0]

        this_gini, this_split_vaule = Cart.gini_index(attr_values, cur_labels, Cart.is_number(tmp_data[title]))
        title_gini[title] = this_gini
        title_split_value[title] = this_split_vaule

    best_attr = min(title_gini, key=title_gini.get) # gini指数最小的属性
    current_node.attr_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if Cart.is_number(a_data[best_attr]): # 如果最佳分割属性为连续值
        split_value = title_split_value[best_attr]
        small_data = []
        big_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if this_data[best_attr]<=split_value:
                small_data.append(index)
            else:
                big_data.append(index)
        small_str = '<='+str(split_value)
        large_str = '>'+str(split_value)
        small_child = TreeNode.TreeNode(parent=current_node,data_index=small_data,attr_value=small_str,rest_attr=rest_title.copy())
        large_child = TreeNode.TreeNode(parent=current_node,data_index=big_data,attr_value=large_str,rest_attr=rest_title.copy())

        # 需要先给子节点一个判断
        small_data_count = {}
        for index in small_child.data_index:
            if small_data_count.__contains__(label[index]):
                small_data_count[label[index]] += 1
            else:
                small_data_count[label[index]] = 1
        small_child_judge = max(small_data_count, key=small_data_count.get)
        small_data.judge = small_child_judge  # 临时判断

        large_data_count = {}
        for index in large_child.data_index:
            if large_data_count.__contains__(label[index]):
                large_data_count[label[index]] += 1
            else:
                large_data_count[label[index]] = 1
        large_data_judge = max(large_data_count, key=large_data_count.get)
        large_child.judge = large_data_judge # 临时判断

        current_node.children = [small_child,large_child]

    else: # 最佳分割属性为离散值
        best_titlevalue_dict = {} # key是属性值的取值，value是list记录所包含的样本序号
        for index in current_node.data_index:
            this_data = data[index]
            if best_titlevalue_dict.__contains__(this_data[best_attr]):
                tmp_list = best_titlevalue_dict[this_data[best_attr]]
                tmp_list.append(index)
            else:
                tmp_list = [index]
                best_titlevalue_dict[this_data[best_attr]] = tmp_list
        children_list = []
        for key, index_list in best_titlevalue_dict.items():
            a_child = TreeNode.TreeNode(parent=current_node, data_index=index_list, attr_value=key,rest_attr=rest_title.copy())

            # 需要给子节点一个判断
            tmp_data_count = {}
            for index in index_list:
                if tmp_data_count.__contains__(label[index]):
                    tmp_data_count[label[index]] += 1
                else:
                    tmp_data_count[label[index]] = 1
            tmp_child_judge = max(tmp_data_count,key=tmp_data_count.get)
            a_child.judge = tmp_child_judge # 临时添加一个判断

            children_list.append(a_child)
        current_node.children = children_list

    current_node.judge = None
    later_acc = current_accuracy(current_node, test_data, test_label)  # 剪枝的正确率
    if before_acc > later_acc:
        # 不剪枝比剪枝的正确率大，则该节点标记为叶结点
        current_node.children = None
        current_node.judge = before_judge
        return
    else:
        for chi in current_node.children:
            finish_node(chi, data, label, test_data, test_label)


def precut_cart_tree(data, title, label, test_data, test_label):
    '''
    生成一颗预剪枝的CART决策树
    :param data: 训练数据集，每一个样本都是一个dict{属性名：属性值,...}，整个dataset是一个list
    :param title: 每个属性的名字
    :param label: 每个训练集样本的类别
    :param test_data: 测试集
    :param test_label: 测试集label
    :return:
    '''
    n = len(data)
    rest_title = title.copy()
    root_data = []
    for i in range(n):
        root_data.append(i)

    root_node = TreeNode.TreeNode(data_index=root_data, rest_attr=rest_title)
    finish_node(root_node, data, label, test_data, test_label)

    return root_node


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

    decision_tree = precut_cart_tree(train_data, title, train_lable, test_data, test_label)
    print('预剪枝之后的决策树是：')
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
    run_test()->precut_cart_tree()->finish_node()->current_accuracy(->classify_data())->gini_index(->is_number()
                                                                                                   ->gini_index_basic(->gini()))->current_accuracy()
              ->print_tree()
              ->classify_data()
    '''
    run_test()