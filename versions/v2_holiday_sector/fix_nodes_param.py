import re

# 读取文件
with open('train_federated_pro_full.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 main() 函数中的 parser.add_argument 部分
# 在 --nodes_start 后面添加 --nodes 参数
old_code = '''    parser.add_argument('--nodes_start', type=int, default=8001)
    parser.add_argument('--nodes_end', type=int, default=8025)'''

new_code = '''    parser.add_argument('--nodes', type=str, default=None, help='节点列表，如: 8001,8005,8013')
    parser.add_argument('--nodes_start', type=int, default=8001)
    parser.add_argument('--nodes_end', type=int, default=8025)'''

content = content.replace(old_code, new_code)

# 找到 config.nodes 赋值的地方，修改逻辑
old_assign = '''    config.nodes = list(range(args.nodes_start, args.nodes_end))'''
new_assign = '''    if args.nodes:
        config.nodes = [int(n.strip()) for n in args.nodes.split(',')]
    else:
        config.nodes = list(range(args.nodes_start, args.nodes_end))'''

content = content.replace(old_assign, new_assign)

# 写回文件
with open('train_federated_pro_full.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 修改完成！现在支持 --nodes 参数")
