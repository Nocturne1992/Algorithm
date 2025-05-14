### 1. Binary Search
- 寻找两个连续的互斥区域的边界
```java
int left = -1;
int right = arr.length;
while (left + 1 != right) {
	int mid = left + (right - left) / 2;
	if (check<arr[mid]>) {
		left = mid;
	} else {
		right = mid;
	}
}

return left;
```
- 寻找某个数
```java
int left = 0;
int right = arr.length - 1;
while (left <= right) {
	int mid = left + (right - left) / 2;
	if (arr[mid] == target) {
		return mid;
	}

	if (arr[mid] < target) {
		left = mid + 1;
	} else {
		right = mid - 1;
	}
}
``` 

### 2. Prefix Sum
- 查看以当前位置为结尾的子数组和的性质
```java
int sum = 0;
for (int i=0; i<arr.length; i++) {
	sum += arr[i];
	<check if the sum of specific subarray ending at i satisfies certain condition>
	<update current prefix sum to the record with hashmap or something else>
}
```