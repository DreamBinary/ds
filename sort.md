堆排序

```c++
#include <bits/stdc++.h>

using namespace std;

inline void HeapAdjust(int *arr, int start, int end) {
    int tmp = arr[start];
    for (int i = 2 * start + 1; i <= end; i = i * 2 + 1) {
        if (i < end && arr[i] < arr[i + 1]) {
            i++;
        }
        if (arr[i] > tmp) {
            arr[start] = arr[i];
            start = i;
        } else {
            break;
        }
    }
    arr[start] = tmp;
}

inline void HeapSort(int *arr, int len) {
    for (int i = (len - 1 - 1) / 2; i >= 0; i--) {
        HeapAdjust(arr, i, len - 1);
    }
    int tmp;
    for (int i = 0; i < len - 1; i++) {
        tmp = arr[0];
        arr[0] = arr[len - 1 - i];
        arr[len - 1 - i] = tmp;
        HeapAdjust(arr, 0, len - 1 - i - 1);
    }
}

int main() {
    int arr[] = {9, 5, 6, 3, 5, 3, 1, 0, 96, 66};
    HeapSort(arr, sizeof(arr) / sizeof(arr[0]));
    for (int i: arr) {
        printf("%d ", i);
    }
    return 0;
}
```

```c++
// h[N]存储堆中的值, h[1]是堆顶，x的左儿子是2x, 右儿子是2x + 1
// ph[k]存储第k个插入的点在堆中的位置
// hp[k]存储堆中下标是k的点是第几个插入的
int h[N], ph[N], hp[N], size;

// 交换两个点，及其映射关系
void heap_swap(int a, int b) {
    swap(ph[hp[a]], ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u) {
    int t = u;
    if (u * 2 <= size && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= size && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t) {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u) {
    while (u / 2 && h[u] < h[u / 2]) {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

// O(n)建堆
for (int i = n / 2; i; i--) {
	down(i);
}
```

并归排序

```c++
#include<iostream>

using namespace std;

void Merge(int arr[], int low, int mid, int high) {
    //low为第1有序区的第1个元素，i指向第1个元素, mid为第1有序区的最后1个元素
    int i = low, j = mid + 1, k = 0; //mid+1为第2有序区第1个元素，j指向第1个元素
    int *temp = new(nothrow) int[high - low + 1]; //temp数组暂存合并的有序序列
    if (!temp) { //内存分配失败
        cout << "error";
        return;
    }
    while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) //较小的先存入temp中
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }`
    while (i <= mid)//若比较完之后，第一个有序区仍有剩余，则直接复制到t数组中
        temp[k++] = arr[i++];
    while (j <= high)//同上
        temp[k++] = arr[j++];
    for (i = low, k = 0; i <= high; i++, k++)//将排好序的存回arr中low到high这区间
        arr[i] = temp[k];
    delete[]temp;//删除指针，由于指向的是数组，必须用delete []
}

//用递归应用二路归并函数实现排序——分治法
void MergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        MergeSort(arr, low, mid);
        MergeSort(arr, mid + 1, high);
        Merge(arr, low, mid, high);
    }
}

int main() {
    int a[10] = {5, 1, 9, 3, 7, 4, 8, 6, 2, 0};
    MergeSort(a, 0, 9);
    for (int i = 0; i < 10; i++)
        cout << a[i] << " ";
    return 0;
}
```

快选

```c++
#include "bits/stdc++.h"
using namespace std;
class Solution0 {
public:
    int quickSelect(vector<int> &a, int l, int r, int index) {
        int q = randomPartition(a, l, r);
        if (q == index) {
            return a[q];
        } else {
            return q < index ? quickSelect(a, q + 1, r, index) : quickSelect(a, l, q - 1, index);
        }
    }

    inline int randomPartition(vector<int> &a, int l, int r) {
        int i = rand() % (r - l + 1) + l;
        swap(a[i], a[r]);
        return partition(a, l, r);
    }

    inline int partition(vector<int> &a, int l, int r) {
        int x = a[r], i = l - 1;
        for (int j = l; j < r; ++j) {
            if (a[j] <= x) {
                swap(a[++i], a[j]);
            }
        }
        swap(a[i + 1], a[r]);
        return i + 1;
    }

    int findKthLargest(vector<int> &nums, int k) {
        srand(time(0));
        return quickSelect(nums, 0, nums.size() - 1, nums.size() - k);
    }
};

class Solution1 {
private:
    mt19937 gen{random_device{}()};
public:
    void quickSelect(vector<vector<int>> &points, int l, int r, int k) {
        int q = uniform_int_distribution<int>{l, r}(gen);
        swap(points[q], points[r]);
        int qq = points[r][0] * points[r][0] + points[r][1] * points[r][1];
        int i = l - 1, j = l;
        while (j < r) {
            if (points[j][0] * points[j][0] + points[j][1] * points[j][1] <= qq) {
                swap(points[j], points[++i]);
            }
            j++;
        }
        swap(points[++i], points[r]);
        if (i < k + l - 1) {
            quickSelect(points, i + 1, r, k - i + l - 1);
        } else if (i - l + 1 > k) {
            quickSelect(points, l, i - 1, k);
        }
    }

    vector<vector<int>> kClosest(vector<vector<int>> &points, int k) {
        int n = points.size();
        quickSelect(points, 0, n - 1, k);
        return {points.begin(), points.begin() + k};
    }
};
int main() {
    vector<int> nums = {3, 2, 1, 5, 6, 4};
    int k = 3;
    nth_element(nums.begin(), nums.begin() + k - 1, nums.end());
    cout << nums[k - 1] << endl; // 3
}
```

双关键字计数排序

```c++
#include "bits/stdc++.h"

using namespace std;
const int maxn = 10000000;
const int maxs = 10000;
int n;
int a[maxn], b[maxn], res[maxn], ord[maxn];
int cnt[maxs + 1];

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; ++i) {
        scanf("%d %d", &a[i], &b[i]);
    }
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < n; ++i) {
        cnt[b[i]]++;
    }
    for (int i = 0; i < maxs; ++i) {
        cnt[i + 1] += cnt[i];
    }
    for (int i = 0;i < 10; ++i) {
        cout << cnt[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < n; ++i) {
        ord[--cnt[b[i]]] = i;
    }
    for (int i = 0; i < 10; ++i) {
        cout << ord[i] << endl;
    }
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < n; ++i) {
        cnt[a[i]]++;
    }
    for (int i = 0; i < maxs; ++i) {
        cnt[i + 1] += cnt[i];
    }
    for (int i = n - 1; i >= 0; --i) {
        res[--cnt[a[ord[i]]]] = ord[i];
    }
    for (int i = 0; i < n; ++i) {
        printf("%d %d\n", a[res[i]], b[res[i]]);
    }
}

//5、计数排序是⼀个⼴泛使用的排序算法。以下算法使
//用双关键字计数排序，将n对整数对进⾏从小到⼤的排
//序，先对第⼆关键字排序，再对第⼀关键字排序。例
//如有三对整数（3，4）、（2，4）、（3，3），那么
//排序以后为（2，4）、（3，3）、（3，4）
//已知输⼊时，整数对成对输⼊并存放在数组单元a[i]和
//b[i]中，其中a[i]表示第i对整数对的第⼀关键字，⽽b[i]
//表示第⼆关键字。整数对的数量不超过maxn，⽽任意
//关键字的取值不超过maxs，即1<=a[i],b[i]<=maxs。
//数组ord []存储第⼆关键字排序结果，数组res[]存储双
//关键字排序结果，数组cnt[]为计数数组。
//请在空行处填写正确的语句。

//6
//1 2
//1 5
//4 8
//3 8
//8 9
//6 7
```

