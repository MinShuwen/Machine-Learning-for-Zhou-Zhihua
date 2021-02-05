'''
利用信息增益作为选择最优属性的标准构建ID3树
计算公式：
Ent(D) = -sigma(p*log_2(p))
连续属性的信息增益：Gain(D,a) = max Gain(D,a,t)
                             = max Ent(D,a) - (λ=-)(|Dt|/|D| * Ent(Dt)) - (λ=+)(|Dt|/|D| * Ent(Dt))
离散属性的信息增益：Gain(D,a) = Ent(D) - sigma(|D_v|/|D| * Ent(D_v))
'''
import math
import DataSet
from TreeNode import TreeNode

def is_number(s):
    '''
    判断属性是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def ent(labels):
    '''
    样本集合的信息熵
    :param labels: 样本集合中数据的类别标签
    :return: Ent(D) = -sigma(p*log_2(p))
    '''
    label_name = [] # labels中每一种label的名称
    label_count = [] # 对应label的数量

    for item in labels:
        if not (item in label_name):
            label_name.append(item)
            label_count.append(1)
        else:
            index = label_name.index(item)
            label_count[index] += 1

    n = sum(label_count)
    entropy = 0.0
    for item in label_count:
        p = item/n
        entropy -= p*math.log(p, 2)
    return entropy

def gain(attribute, labels, is_value = False):
    '''
    计算每一个属性对应的信息增益
    :param attribute: 集合中样本该属性的值列表，等价于data数据中的列
    :param labels: 集合中样本的数据标签
    :param is_value: 属性是离散值还是连续值
    :return: 连续属性的信息增益：Gain(D,a) = max Gain(D,a,t)
                                          = max Ent(D,a) - (λ=-)(|Dt|/|D| * Ent(Dt)) - (λ=+)(|Dt|/|D| * Ent(Dt))
    离散属性的信息增益：Gain(D,a) = Ent(D) - sigma(|D_v|/|D| * Ent(D_v))
    '''
    info_gain = ent(labels)
    n = len(labels)
    split_value = None # 如果是连续值，需要返回分割界限的值

    if is_value:
        # print('attribute', attribute)
        # 属性值是连续值，首先应该使用二分法寻找最佳分割点
        sorted_attribute = attribute.copy()
        sorted_attribute.sort()
        split = [] # 候选分割点，二分法，将所有属性取值排序后选择相邻两点的平均数
        for i in range(0,n-1):
            tmp = (sorted_attribute[i]+sorted_attribute[i+1])/2
            split.append(tmp)
        info_gain_list = []
        for tmp_split in split:
            low_labels = []
            high_labels = []
            for i in range(0,n):
                if attribute[i]<=tmp_split:
                    low_labels.append(labels[i])
                else:
                    high_labels.append(labels[i])
            tmp_gain = info_gain - len(low_labels)/n*ent(low_labels) - len(high_labels)/n*ent(high_labels)
            info_gain_list.append(tmp_gain)
        # print('info_gain_list : ', info_gain_list)
        info_gain = max(info_gain_list)
        max_index = info_gain_list.index(max(info_gain_list))
        split_value = split[max_index]
    else:
        # 属性值是离散值
        attribute_dict = {} # 属性a取v值时，对应样本数量Dv 例如：{'青绿':6,'乌黑':6,'浅白':5}
        label_dict = {} # 属性a取v值时，对应样本的label 例如{'青绿':['好瓜','好瓜','好瓜',...],'乌黑':['好瓜','好瓜','好瓜',...],'浅白':['好瓜','坏瓜','坏瓜',...]}
        index = 0
        for item in attribute:
            if attribute_dict.__contains__(item):
                attribute_dict[item] += 1
                label_dict[item].append(labels[index])
            else:
                attribute_dict[item] = 1
                label_dict[item] = [labels[index]]
            index += 1

        for key, value in attribute_dict.items():
            info_gain -= value/n*ent(label_dict[key])

    return info_gain, split_value

def finish_node(current_node, data, label):
    '''
    完成当前节点的后续计算，包括选择属性，划分子节点等
    :param current_node: 当前节点 TreeNode
    :param data: 数据集 {list}: [{attr_name:attr_value,...},{},{},...]
    :param label: 数据集的标签
    :param rest_title: 剩余的可用属性名
    :return:
    '''
    n = len(label)

    # 判断当前节点的数据是否属于同一类，如果是，直接标记为叶子节点并返回
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

    rest_title = current_node.rest_attr # 候选属性
    if len(rest_title) == 0:
        # 如果候选属性为空，则是个叶子节点，选择最多的那个类作为该节点的类
        label_count = {}
        tmp_data = current_node.data_index
        for index in tmp_data:
            if label_count.__contains__(label[index]):
                label_count[label[index]] += 1
            else:
                label_count[label[index]] = 1
        final_label = max(label_count,key=label_count.get)
        current_node.judge = final_label
        return

    title_gain = {} # 记录每个属性的信息增益
    title_split_value = {} # 记录每个属性的分割值，如果是连续属性则为分割值，如果是离散属性则为None
    for title in rest_title: # 从rest_title中的属性中选择最佳分割属性
        attr_values = []
        current_label = []
        for index in current_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            current_label.append(label[index])
        tmp_data = data[0]
        this_gain, this_split_value = gain(attr_values,current_label,is_number(tmp_data[title]))
        title_gain[title] = this_gain
        title_split_value[title] = this_split_value
    best_attr = max(title_gain, key=title_gain.get) # 信息增益最大的属性名
    current_node.attr_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if is_number(a_data[best_attr]):
        # 如果该属性的值为连续值
        split_value = title_split_value[best_attr]
        small_data = []
        large_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if this_data[best_attr]<=split_value:
                small_data.append(index)
            else:
                large_data.append(index)
        small_str = '<=' + str(small_data)
        large_str = '>' + str(large_data)
        small_child = TreeNode(parent=current_node, data_index=small_data, attr_value=small_str,
                               rest_attr= rest_title.copy())
        large_child = TreeNode(parent=current_node, data_index=large_data,attr_value=large_str,
                               rest_attr= rest_title.copy())
        current_node.children = [small_child, large_child]
    else:
        # 该属性是离散值
        best_titlevalue_dict = {} # key:属性值的取值，value:list,记录所包含的样本序号
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
            a_child = TreeNode(parent=current_node, data_index=index_list, attr_value = key,
                               rest_attr=rest_title.copy())
            children_list.append(a_child)
        current_node.children = children_list
     # print(current_node.to_string())

    for child in current_node.children: # 递归
        finish_node(child, data, label)

def id3_tree(Data, title, label):
    '''
    id3方法构建决策树，使用的标准是信息增益
    :param Data: 数据集，每一个样本是一个dict(属性名：属性值)，整个dataset是一个大的list
    :param title: 每个属性的名字，如 色泽、含糖量 等
    :param label: 存储的是每个样本的类别
    :return:
    '''
    n = len(Data) # 17
    rest_title = title.copy()
    root_data = []
    for i in range(0,n):
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

def run_test():
    watermelon, title, title_full = DataSet.watermelon()
    # watermelon = {list} [[],[],...]
    # title = {list} ['色泽','根蒂',...]
    # title_full = {dict} {'色泽':{青绿,浅白,乌黑},'根蒂'：{},...}

    # 先处理数据结构
    data = [] # 存放数据
    label = [] # 存档标签
    for melon in watermelon:
        a_dict = {}
        dim = len(melon)-1
        for i in range(0,dim):
            a_dict[title[i]] = melon[i]
        data.append(a_dict)
        label.append(melon[dim])
    decision_tree = id3_tree(data, title, label)
    # data:{list} [{'色泽':乌黑,'根蒂':蜷缩,'敲击':沉闷,...},{},...]
    # title = {list} ['色泽','根蒂',...]
    # label:{list} ['好瓜','好瓜',...]
    print_tree(decision_tree)

if __name__ == '__main__':
    '''
    run_test()->id3_tree()->finish_node()->gain(->is_number())->ent()
              ->print_tree()
    '''
    run_test()
