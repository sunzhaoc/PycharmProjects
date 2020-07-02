# 定义字符串变量name，输出 我的名字叫 小名，请多多关照！
name = "小明"
print("我的名字叫 %s，请多多关照！" %name)

# 定义整数变量student_no，输出 我的学号是 000001
student_no = 1
print("我的学号是 %06d" %student_no)

# 定义小数price, weight, money
# 输出 苹果单价 9.00元/斤，购买了5.00斤，需要支付45.00元
price = 8.5
weight = 7.5
money = price * weight

print("苹果单价 %.2f元/斤，购买了%.3f斤，需要支付%.4f元" %(price, weight, money))

# 定义一个小数scale，输出数据比例是10。00%
scale = 0.25
print("数据比例是 %.2f%%" %(scale*100))
