from utility import set_counter

print("Enter the name of the test file")
test_name = input()
if not test_name.endswith(".csv"):
	test_name = test_name + ".csv"
with open("../data/utilities/test_name.txt", 'w') as f:
    f.write(test_name)

print('Enter the line you want to start the testing from:')
n = int(input())
n = 1 if n < 1 else n
set_counter(n)
print("test_name set to " + test_name + ", counter set to " + str(n))
