class TreeNode:
    '''
    决策树节点类
    '''
    current_index = 0
    def __init__(self, parent = None, attr_name = None, children = None, judge = None,
                 split = None, data_index = None, attr_value = None, rest_attr = None):
        self.parent = parent # 父节点，根节点的父节点为None
        self.attr_name = attr_name # 本节点上进行划分的属性名
        self.attr_value = attr_value # 本节点划分属性的值，与父节点的划分属性名相对应
        self.children = children # 孩子节点列表
        self.judge = judge # 如果是叶子节点，给出判断结果
        self.split = split # 如果使用连续属性进行划分，需要给出分割点
        self.data_index = data_index # 对应训练数据集的训练索引号，为数组
        self.index = TreeNode.current_index # 当前节点的索引号，方便输出时查看，为单个值
        self.rest_attr = rest_attr # 尚未使用的属性列表
        TreeNode.current_index += 1

    def to_string(self):
        '''用一个字符串来描述当前节点信息'''
        this_string = 'current index : '+str(self.index) + ';\n'
        # current index : 2
        if not(self.parent is None):
            parent_node = self.parent
            this_string += 'parent index : ' + str(parent_node.index) + ';\n'
            this_string += str(parent_node.attr_name) + ' : ' + str(self.attr_value) + ';\n'
        this_string += 'data : ' + str(self.data_index) + ';\n'
        # parent index : 1;
        # 纹理 ：清晰；
        # data : [0,1,2,3,4,5,7,9,14]
        if not(self.children is None):
            this_string += 'select attribute is : ' + str(self.attr_name) + ';\n'
            child_list = []
            for child in self.children:
                child_list.append(child.index)
            this_string += 'children : '+ str(child_list)
        # select attribute is : 密度；
        # children : [5,6]
        if not (self.judge is None):
            this_string += 'label : ' + self.judge
        # label : 坏瓜
        return this_string