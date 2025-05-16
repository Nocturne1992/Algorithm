### 1. Binary Search
- 寻找两个连续的互斥区域的边界
```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = -1;
        int right = nums.length;

        while (left + 1 != right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid;
            }
        }

        return right;
    }
}
```
- 左右边界包含了可能的答案
```python
class Solution {
    public int findPeakElement(int[] nums) {
        int left = -1;
        int right = nums.length;

        while (left + 1 != right) {
            int mid = left + (right - left) / 2;

            if (isUp(nums, mid)) {
                left = mid;
            } else {
                right = mid;
            }
        }

        return right;
    }

    public boolean isUp(int[] nums, int idx) {
        return idx + 1 < nums.length && nums[idx] < nums[idx + 1];
    }
}
```
- 寻找某个数
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            }

            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }
}
``` 

### 2. Prefix Sum
- 前缀和数组
```java
class NumArray {
    int[] pre;
    public NumArray(int[] nums) {
        pre = new int[nums.length + 1];
        for (int i=1; i<pre.length; i++) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
    }
    
    public int sumRange(int left, int right) {
        return pre[right + 1] - pre[left];
    }
}
```
- 前缀后缀和
```java
class Solution {
    public long maximumSumScore(int[] nums) {
        long[] pre = new long[nums.length];
        pre[0] = nums[0];

        for (int i=1; i<pre.length; i++) {
            pre[i] = pre[i - 1] + nums[i];
        }

        long[] suf = new long[nums.length];
        suf[nums.length - 1] = nums[nums.length - 1];

        for (int i=suf.length - 2; i>=0; i--) {
            suf[i] = suf[i + 1] + nums[i];
        }

        long max = Long.MIN_VALUE;

        for (int i=0; i<nums.length; i++) {
            max = Math.max(max, Math.max(pre[i], suf[i]));
        }

        return max;
    }
}
```
- 查看以当前位置为结尾的子数组和的性质
```java

```