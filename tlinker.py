import tkinter
from tkinter import ttk

win = tkinter.Tk()
win.title("Kahn Software v1")  # #窗口标题
win.geometry("600x500+200+20")  # #窗口位置500后面是字母x
'''
树状结构
'''
tree = ttk.Treeview(win)  # #创建树状对象
# #一级目录
treeF1 = tree.insert("", 0, "上海", text="上海SH", values=("F1"))  # #创建一级树目录
treeF2 = tree.insert("", 1, "江苏", text="江苏JS", values=("F2"))
treeF3 = tree.insert("", 2, "浙江", text="浙江ZJ", values=("F3"))
# #二级目录
treeF1_1 = tree.insert(treeF1, 0, "黄浦区", text="黄浦区hp", values=("F1_1"))  # #将目录帮到菜单treeF1
treeF1_2 = tree.insert(treeF1, 1, "静安区", text="静安区ja", values=("F1_2"))
treeF1_3 = tree.insert(treeF1, 2, "长宁区", text="长宁区cn", values=("F1_3"))
treeF2_1 = tree.insert(treeF2, 0, "苏州", text="苏州sz", values=("F2_1"))  # #将目录帮到菜单treeF2
treeF2_2 = tree.insert(treeF2, 1, "南京", text="南京nj", values=("F2_2"))
treeF2_3 = tree.insert(treeF2, 2, "无锡", text="无锡wx", values=("F2_3"))
treeF3_1 = tree.insert(treeF3, 0, "杭州", text="杭州hz", values=("F3_1"))  # #将目录帮到菜单treeF3
treeF3_2 = tree.insert(treeF3, 1, "宁波", text="宁波nb", values=("F3_2"))
treeF3_3 = tree.insert(treeF3, 2, "温州", text="温州wz", values=("F3_3"))
# #三级目录
treeF1_1_1 = tree.insert(treeF1_1, 0, "南京路", text="南京路njl", values=("treeF1_1_1"))
treeF1_1_2 = tree.insert(treeF1_1, 0, "河南路", text="河南路hnl", values=("treeF1_1_2"))
treeF1_1_3 = tree.insert(treeF1_1, 0, "延安路", text="延安路yal", values=("treeF1_1_3"))
tree.pack()
win.mainloop()
