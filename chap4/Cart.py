'''
利用gini系数选择最优划分属性，
反映了从数据集D中随机抽取来两个样本，这两个样本不属于同一类的概率，
值越小越好
计算公式：
Gini(D) = 1-sigma(p_k^2)
Gini_index(D,a) = sigma(Dv/D*Gini(Dv))
'''
from TreeNode import TreeNode
import math
import DataSet


def is_number(s):
    '''判断一个字符串是否是数字，如果是数字，则对应的属性为连续值'''
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def gini(labels=None):
    '''
    计算数据集的基尼值
    :param labels: 数据集的类别标签
    :return:
    '''
    if labels is None:
        labels = []
    data_count = {}
    for label in labels:
        if data_count.__contains__(label):
            data_count[label] += 1
        else:
            data_count[label] = 1
    n = len(labels)
    if n == 0:
        return 0
    gini_value = 1
    for key, value in data_count.items():
        gini_value -= (value / n) * (value / n)
    return gini_value


def gini_index_basic(n, attr_labels=None):
    gini_value = 0
    for attr, labels in attr_labels.items():
        gini_value += len(labels) / n * gini(labels)
    return gini_value


def gini_index(attrs = None, labels = None, is_value = False):
    '''
    计算一个属性的gini指数
    :param attrs: 当前数据在该属性上的属性值列表
    :param labels: 数据的标签
    :param is_value: 是否为连续值属性
    :return:
    '''
    n = len(labels)
    attr_labels = {}
    gini_value = 0 # 最终返回结果
    split = None

    if is_value:
        # 属性值是连续的数
        sorted_attrs = attrs.copy()
        sorted_attrs.sort()
        split_points = []
        for i in range(n-1):
            split_points.append((sorted_attrs[i]+sorted_attrs[i+1])/2)
        split_eval = []
        for cur_split in split_points:
            low_labels = []
            high_labels = []
            for i in range(n):
                if attrs[i]<=cur_split:
                    low_labels.append(labels[i])
                else:
                    high_labels.append(labels[i])
            attr_labels = {'small':low_labels, 'large':high_labels}
            split_eval.append(gini_index_basic(n,attr_labels=attr_labels))
        gini_value = min(split_eval)
        split = split_points[split_eval.index(gini_value)]
    else:
        # 属性值为离散值
        for i in range(n):
            if attr_labels.__contains__(attrs[i]):
                tmp_list = attr_labels[attrs[i]]
                tmp_list.append(labels[i])
            else:
                tmp_list = []
                tmp_list.append(labels[i])
                attr_labels[attrs[i]] = tmp_list
        gini_value = gini_index_basic(n,attr_labels=attr_labels)

    return gini_value, split


def finish_node(cur_node = TreeNode(), data = None, labels = None):
    '''
    完成一个节点上的计算
    :param cur_node: 当前计算的节点
    :param data: 数据集
    :param labels: 数据集的label
    :return:
    '''
    n = len(labels)

    # 判断当前节点中的数据是否属于同一类，如果为同一类，则直接标记为叶子节点
    one_class = True
    this_data_index = cur_node.data_index

    for i in this_data_index:
        for j in this_data_index:
            if labels[i] != labels[j]:
                one_class = False
                break
        if not one_class:
            break
    if one_class:
        cur_node.judge = labels[this_data_index[0]]
        return

    rest_title = cur_node.rest_attr # 候选属性
    if len(rest_title)==0:
        # 如果候选属性为空，则标记为叶子节点，选择最多的那个类别作为该节点的类
        label_count = {}
        tmp_data = cur_node.data_index
        for index in tmp_data:
            if label_count.__contains__(labels[index]):
                label_count[labels[index]] += 1
            else:
                label_count[labels[index]] = 1
        final_label = max(label_count,key=label_count.get)
        # max(dict,key=dict.get) 则求出来的最大值为 最大的value对应的key
        cur_node.judge = final_label
        return

    title_gini = {} # 记录每个属性的gini指数
    title_split_value = {} # 记录每个属性的分割值，如果是连续属性，则为分割值，如果是离散属性，则为None
    for title in rest_title:
        attr_values = []
        cur_labels = []
        for index in cur_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            cur_labels.append(labels[index])
        tmp_data = data[0]
        this_gini, this_split_vaule = gini_index(attr_values, cur_labels, is_number(tmp_data[title]))
        title_gini[title] = this_gini
        title_split_value[title] = this_split_vaule

    best_attr = min(title_gini, key=title_gini.get) # gini指数最小的属性
    cur_node.attr_name = best_attr
    cur_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if is_number(a_data[best_attr]): # 如果最佳分割属性为连续值
        split_value = title_split_value[best_attr]
        small_data = []
        big_data = []
        for index in cur_node.data_index:
            this_data = data[index]
            if this_data[best_attr]<=split_value:
                small_data.append(index)
            else:
                big_data.append(index)
        small_str = '<='+str(split_value)
        large_str = '>'+str(split_value)
        small_child = TreeNode(parent=cur_node,data_index=small_data,attr_value=small_str,rest_attr=rest_title.copy())
        large_child = TreeNode(parent=cur_node,data_index=big_data,attr_value=large_str,rest_attr=rest_title.copy())
        cur_node.children = [small_child,large_child]

    else: # 最佳分割属性为离散值
        best_titlevalue_dict = {} # key是属性值的取值，value是list记录所包含的样本序号
        for index in cur_node.data_index:
            this_data = data[index]
            if best_titlevalue_dict.__contains__(this_data[best_attr]):
                tmp_list = best_titlevalue_dict[this_data[best_attr]]
                tmp_list.append(index)
            else:
                tmp_list = [index]
                best_titlevalue_dict[this_data[best_attr]] = tmp_list
        children_list = []
        for key, index_list in best_titlevalue_dict.items():
            a_child = TreeNode(parent=cur_node, data_index=index_list, attr_value=key,rest_attr=rest_title.copy())
            children_list.append(a_child)
        cur_node.children = children_list

    for chi in cur_node.children:
        finish_node(chi,data,labels)


def cart_tree(Data, title, label):
    '''
    生成一颗 CART 决策树
    :param Data: 数据集，每一个样本都是一个dict（属性名：属性值），整个Data是一个list
    :param title: 每个属性的名字，如色泽、含糖量等
    :param label: 每个样本的类别
    :return:
    '''
    n = len(Data)
    rest_title = title.copy()
    root_data = []
    for i in range(n):
        root_data.append(i)
    root_node = TreeNode(data_index=root_data, rest_attr=rest_title)
    finish_node(root_node, Data, label)
    return root_node


def print_tree(root=TreeNode()):
    '''
    d打印输出一棵树
    :param root: 根节点
    :return:
    '''
    node_list = [root]
    while len(node_list)>0:
        cur_node = node_list[0]
        print('-------------------')
        print(cur_node.to_string())
        print('-------------------')
        children_list = cur_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(cur_node)


def classify_data(decision_tree = TreeNode(), x = None):
    '''
    使用决策树判断一个数据样本的类别标签
    :param decision_tree: 决策树的根节点
    :param x: 要进行判断的样本
    :return:
    '''
    cur_node = decision_tree
    while cur_node.judge is None:
        if cur_node.split is None: # 离散属性
            can_judge = False # 可能存在测试数据中出现训练数据中没有出现过的属性值
            for child in cur_node.children:
                if child.attr_value == x[cur_node.attr_name]:
                    cur_node = child
                    can_judge = True
                    break
            if not can_judge:
                return None
        else:
            child_list = cur_node.children
            if x[cur_node.attr_name] <= cur_node.split:
                cur_node = child_list[0] # 进入左子树
            else:
                cur_node = child_list[1] # 进入右子树

    return cur_node.judge


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

    decision_tree = cart_tree(train_data, title, train_lable)
    print('训练的决策树是：')
    print_tree(decision_tree)
    print('\n')

    test_judge = []
    for melon in test_data:
        test_judge.append(classify_data(decision_tree,melon))
    print('决策树在测试数据集上的分类结果是：',test_judge)
    print('测试数据集的正确类别信息是：',test_label)

    acc = 0
    for i in range(len(test_label)):
        if test_label[i] == test_judge[i]:
            acc += 1
    acc /= len(test_label)
    print('决策树在测试数据集上的分类正确率为：',str(acc*100)+'%')

'''
决策树在测试数据集上的分类结果是： ['否', '否', '否', '是', '否', '否', '是']
测试数据集的正确类别信息是： ['是', '是', '是', '否', '否', '否', '否']
决策树在测试数据集上的分类正确率为： 28.57142857142857%
'''


if __name__ == '__main__':
    '''
    run_test()->cart_tree()->finish_node()->gini_index(->is_number())->gini_index_basic()->gini()
              ->print_tree()
              ->classify_data()
    '''
    run_test()

