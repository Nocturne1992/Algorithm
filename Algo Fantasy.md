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
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        int sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);

        for (int i=0; i<nums.length; i++) {
            sum += nums[i];

            if (map.containsKey(sum % k) && i - map.get(sum % k) >= 2) {
                return true;
            }

            if (!map.containsKey(sum % k)) {
                map.put(sum % k, i);
            }
        }

        return false;
    }
}
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

### 4. 3Sum
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

### 5. Sliding Window
- 不定长滑动窗口
```java
//前提是问题性质会随着窗口边界单调变化
//在最外层循环[left, right)不满足有重复字符条件
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int left = 0;
        int right = 0;
        int max = 0;
        Map<Character, Integer> map = new HashMap<>();

        while (right < s.length()) {
            char rc = s.charAt(right);
            map.put(rc, map.getOrDefault(rc, 0) + 1);

            while (map.get(rc) > 1) {
                char lc = s.charAt(left);
                map.put(lc, map.get(lc) - 1);
                left++;
            }

            max = Math.max(max, right - left + 1);
            right++;
            
        }

        return max;
    }
}
```

- 定长滑动窗口
```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        Map<Character, Integer> smap = new HashMap<>();
        Map<Character, Integer> rmap = new HashMap<>();

        for (int i=0; i<s1.length(); i++) {
            char c = s1.charAt(i);
            rmap.put(c, rmap.getOrDefault(c, 0) + 1);
        }

        int left = 0;
        int right = 0;
        int match = 0;

        while (right < s2.length()) {
            char rc = s2.charAt(right);

            smap.put(rc, smap.getOrDefault(rc, 0) + 1);

            if (rmap.containsKey(rc) && rmap.get(rc) == smap.get(rc) + 0) {
                match++;
            } 

            while (right - left + 1 > s1.length()) {
                char lc = s2.charAt(left);
                if (rmap.containsKey(lc) && rmap.get(lc) == smap.get(lc) + 0) {
                    match--;
                }
                smap.put(lc, smap.get(lc) - 1);
                left++;         
            }

            if (match == rmap.size()) {
                return true;
            }

            right++;
        }

        return false;
    }
}
```
### 6. Cutting Rope / String
```java
//截取从idx开始最长为k: for (int i=idx; i<idx + k; i++) -> s.substring(idx, i + 1)
class Solution {
    List<String> rst = new ArrayList<>();
    List<String> list = new ArrayList<>();

    public List<String> restoreIpAddresses(String s) {
        dfs(s, 0);
        return rst;
    }

    public void dfs(String s, int idx) {
        if (idx == s.length() && list.size() == 4) {
            rst.add(String.join(".", list));
            return;
        }

        if (idx == s.length() || list.size() == 4) {
            return;
        }

        if (s.charAt(idx) == '0') {
            list.add("0");
            dfs(s, idx + 1);
            list.remove(list.size() - 1);
        } else {
            for (int i=idx; i<Math.min(idx + 3, s.length()); i++) {
                String str = s.substring(idx, i + 1);
                int t = Integer.parseInt(str);
                if (t <= 255) {
                    list.add(str);
                    dfs(s, i + 1);
                    list.remove(list.size() - 1);
                }
            }
        }
    }
}
```

### 7. Palindrome
- 中心开花
```java
class Solution {
    public int countSubstrings(String s) {
        int num = 0;
        for (int i=0; i<s.length(); i++) {
            num += find(s, i, i);
            num += find(s, i , i + 1);
        }

        return num;
    }

    public int find(String s, int left, int right) {
        int num = 0;

        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            num++;
            left--;
            right++;
        }

        return num;
    }
}
```

- 双指针
```java
class Solution {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;

        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }

        return true;
    }
}
```

### 8. Monotonic Stack / Queue
-  Monotonic Stack
```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stk = new ArrayDeque<>();
        int max = 0;

        for (int i=0; i<heights.length; i++) {
            while (!stk.isEmpty() && heights[stk.peekLast()] > heights[i]) {
                int h = heights[stk.pollLast()];
                int right = i - 1;
                int left = stk.isEmpty()? 0 : stk.peekLast() + 1;

                max = Math.max(max, h * (right - left + 1));

            } 

            stk.offerLast(i);
        }

        while (!stk.isEmpty()) {
            int h = heights[stk.pollLast()];
            int right = heights.length - 1;
            int left = stk.isEmpty()? 0 : stk.peekLast() + 1;

            max = Math.max(max, h * (right - left + 1));
        }

        return max;
    }
}
```

- Monotonic Queue
```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> dq = new ArrayDeque<>();

        int left = 0;
        int right = 0;
        int t = 0;

       int[] rst = new int[nums.length - k + 1];

        while (right < nums.length) {
            while (!dq.isEmpty() && nums[dq.peekLast()] < nums[right]){
                dq.pollLast();
            } 
            dq.offerLast(right);

            while (right - left + 1 > k) {
                if (left == dq.peekFirst()) {
                    dq.pollFirst();
                }
                left++;
            }

            if (right - left + 1 == k) {
                rst[t++] = nums[dq.peekFirst()];
            }

            right++;
        }

        return rst;
    }
}
```

```java
class Solution {
    public int longestSubarray(int[] nums, int limit) {
        Deque<Integer> maxq = new ArrayDeque<>();
        Deque<Integer> minq = new ArrayDeque<>();

        int left = 0;
        int right = 0;

        int max = 0;

        while (right < nums.length) {
            while (!maxq.isEmpty() && maxq.peekLast() < nums[right]) {
                maxq.pollLast();
            }
            maxq.offerLast(nums[right]);

            while (!minq.isEmpty() && minq.peekLast() > nums[right]) {
                minq.pollLast();
            }
            minq.offerLast(nums[right]);

            while (maxq.peekFirst() - minq.peekFirst() > limit) {
                if (nums[left] == maxq.peekFirst() + 0) {
                    maxq.pollFirst();
                }

                if (nums[left] == minq.peekFirst() + 0) {
                    minq.pollFirst();
                }

                left++;
            }

            max = Math.max(right - left + 1, max);
            right++;
        }

        return max;
    }
}
```

### Union Find
```java
//Every edge means the connectivity between two nodes, and in union find this connectivity is represented by two nodes share the same root node
class Solution {
    public int countComponents(int n, int[][] edges) {
        UF uf = new UF(n);

        for (int[] edge : edges) {
            uf.connect(edge[0], edge[1]);
        }

        return uf.cnt;
    }
}

class UF{
    int[] pnt;
    int cnt;

    public UF(int n) {
        cnt = n;
        pnt = new int[n];

        for (int i=0; i<pnt.length; i++) {
            pnt[i] = i;
        }
    }

    public int find(int x) {
        if (x != pnt[x]) {
            pnt[x] = find(pnt[x]);
        }

        return pnt[x];
    }

    public void connect(int p, int q) {
        int proot = find(p);
        int qroot = find(q);

        if (proot == qroot) {
            return;
        }

        pnt[proot] = qroot;
        cnt--;
    }

    public boolean isConnected(int p, int q) {
        return find(p) == find(q);
    }
}
```