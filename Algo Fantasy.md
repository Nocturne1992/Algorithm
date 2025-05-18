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

### 3. Sort Method
- Rainbow Sort
```java
class Solution {
    public void sortColors(int[] nums) {
        int i = 0;
        int j = 0;
        int k = nums.length - 1;

        while (j <= k) {
            if (nums[j] == 0) {
                swap(nums, i++, j++);
            } else if (nums[j] == 1) {
                j++;
            } else {
                swap(nums, j, k--);
            }
        }
    }

    public void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

- Merge Sort
```java
class Solution {
    int[] tmp;

    public int[] sortArray(int[] nums) {
        tmp = new int[nums.length];
        mergeSort(nums, 0, nums.length - 1);
        return nums;
    }

    public void mergeSort(int[] nums, int left, int right) {
        if (left == right) {
            return;
        }

        int mid = left + (right - left) / 2;
        mergeSort(nums, left, mid);
        mergeSort(nums, mid + 1, right);

        merge(nums, left, mid, right);
    }

    public void merge(int[] nums, int left, int mid, int right) {
        for (int i=left; i<=right; i++) {
            tmp[i] = nums[i];
        }

        int i = left;
        int j = mid + 1;
        int k = left;

        while (i <= mid && j <= right) {
            if (tmp[i] <= tmp[j]) {
                nums[k++] = tmp[i++];
            } else {
                nums[k++] = tmp[j++];
            }
        }

        while (i <= mid) {
            nums[k++] = tmp[i++];
        }
    }
}
```

- Quick Sort
```java
class Solution {
    Random rand;
    public int[] sortArray(int[] nums) {
        rand = new Random();
        quickSort(nums, 0, nums.length - 1);
        return nums;
    }

    public void quickSort(int[] nums, int left, int right) {
        if (left >= right) {
            return;
        }

        int pvt = left + rand.nextInt(right - left + 1);
        swap(nums, pvt, right);
        int i = left;
        int j = left;
        int k = right - 1;

        while (j <= k) {
            if (nums[j] < nums[right]) {
                swap(nums, i++, j++);
            } else if (nums[j] == nums[right]) {
                j++;
            } else {
                swap(nums, j, k--);
            }
        }

        swap(nums, j, right);
        quickSort(nums, left, i - 1);
        quickSort(nums, j + 1, right); 
    }

    public void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

### 3 Sum
```java
// 任意解都能被这种方法覆盖 所以不会漏解
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> rst= new ArrayList<>();

        Arrays.sort(nums);
        for (int i=0; i<nums.length; i++) {
            List<List<Integer>> lists = twoSum(nums, i + 1, -nums[i]);

            for (List<Integer> list : lists) {
                list.add(nums[i]);
                rst.add(list);
            }

            while (i + 1 < nums.length && nums[i] == nums[i + 1]) {
                i++;
            }
        }

        return rst;
    }

    public List<List<Integer>> twoSum(int[] nums, int idx, int target) {
        List<List<Integer>> list = new ArrayList<>();
        int left = idx;
        int right = nums.length - 1;

        while (left < right) {
            int total = nums[left] + nums[right];

            if (total > target) {
                right--;
            } else if (total < target) {
                left++;
            } else {
                list.add(new ArrayList(Arrays.asList(nums[left], nums[right])));

                int val = nums[left];
                while(left < right && nums[left] == val) {
                    left++;
                }
            }
        }

        return list;
    }
}
```