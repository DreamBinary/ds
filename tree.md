二叉排序

```c++
#include "bits/stdc++.h"
using namespace std;
struct Node {
    int data;
    Node *left;
    Node *right;
};

// 二叉排序
Node *create(Node *root, int data) {
    if (root == nullptr) {
        return new Node{data, nullptr, nullptr};
    }
    if (data > root->data) {
        root->left = create(root->left, data);
    } else {
        root->right = create(root->right, data);
    }
    return root;
}

void print(string s, int num, Node *root) {
    if (root == nullptr) {
        return;
    }
    print("l", root->data, root->left);
    cout << s << num << " -- " << root->data << endl;
    print("r", root->data, root->right);
}

int main() {
    Node *head = nullptr;
    int a;
    while (cin >> a && a) {
        head = create(head, a);
    }
    print(" ", 0, head);
}
```

倍增 lca

```c++
#include<cstdio>
#include<iostream>
#include<cstring>

using namespace std;
const int maxn = 500000 + 2;
int n, m, s;
int k = 0;
int head[maxn], d[maxn], p[maxn][21];//head数组就是链接表标配了吧？d存的是深度（deep）,p[i][j]存的[i]向上走2的j次方那么长的路径
struct node {
    int v, next;
} e[maxn * 2];//存树
void add(int u, int v) {
    e[k].v = v;
    e[k].next = head[u];
    head[u] = k++;
}               //加边函数
void dfs(int u, int fa) {
    d[u] = d[fa] + 1;
    p[u][0] = fa;
    for (int i = 1; (1 << i) <= d[u]; i++)
        p[u][i] = p[p[u][i - 1]][i - 1];
    for (int i = head[u]; i != -1; i = e[i].next) {
        int v = e[i].v;
        if (v != fa) {
            dfs(v, u);
        }
    }
}                               //首先进行的预处理，将所有点的deep和p的初始值dfs出来
int lca(int a, int b)                                          //非常标准的lca查找
{
    if (d[a] > d[b])
        swap(a, b);           //保证a是在b结点上方，即a的深度小于b的深度
    for (int i = 20; i >= 0; i--)
        if (d[a] <= d[b] - (1 << i))
            b = p[b][i];             //先把b移到和a同一个深度
    if (a == b)
        return a;                 //特判，如果b上来和就和a一样了，那就可以直接返回答案了
    for (int i = 20; i >= 0; i--) {
        if (p[a][i] == p[b][i])
            continue;
        else
            a = p[a][i], b = p[b][i];           //A和B一起上移
    }
    return p[a][0];               //找出最后a值的数字
}

int main() {
    memset(head, -1, sizeof(head));
    int a, b;
    scanf("%d%d%d", &n, &m, &s);
    for (int i = 1; i < n; i++) {
        scanf("%d%d", &a, &b);
        add(a, b);
        add(b, a);                      //无向图，要加两次
    }
    dfs(s, 0);
    for (int i = 1; i <= m; i++) {
        scanf("%d%d", &a, &b);
        printf("%d\n", lca(a, b));
    }
    return 0;
}
```

in, pre to post

```C++
#include "bits/stdc++.h"
using namespace std;
vector<int> pre, in;
void post(int root, int start, int end) {
    if (start > end) return;
    int i = start;
    while (i < end && in[i] != pre[root]) i++;
    post(root + 1, start, i - 1);
    post(root + 1 + i - start, i + 1, end);
    cout << pre[root];
}
```

in, post to pre

```C++
#include "bits/stdc++.h"
using namespace std;
vector<int> post, in;
void pre(int root, int start, int end) {
    if (start > end) return;
    int i = start;
    while (i < end && in[i] != post[root]) i++;
    cout << post[root];
    pre(root - 1 - end + i, start, i - 1);
    pre(root - 1, i + 1, end);
}
```

in, post to level

```c++
#include "bits/stdc++.h"

using namespace std;
vector<int> post, in;
map<int, int> level;

void pre(int root, int start, int end, int index) {
    if (start > end) return;
    int i = start;
    while (i < end && in[i] != post[root]) i++;
    level[index] = post[root];
    pre(root - 1 - (end - i), start, i - 1, 2 * index + 1);
    pre(root - 1, i + 1, end, 2 * index + 2);
}

int main() {
    int n;
    scanf("%d", &n);
    post.resize(n);
    in.resize(n);
    for (int i = 0; i < n; i++) scanf("%d", &post[i]);
    for (int i = 0; i < n; i++) scanf("%d", &in[i]);
    pre(n - 1, 0, n - 1, 0);
    auto it = level.begin();
    printf("%d", it->second);
    while (++it != level.end()) printf(" %d", it->second);
    return 0;
}
```

avl

```c++
#include "bits/stdc++.h"

using namespace std;

struct TreeNode {
    int val;
    TreeNode *right;
    TreeNode *left;
};

inline TreeNode *leftRotate(TreeNode *root) {
    TreeNode *right = root->right;
    root->right = right->left;
    right->left = root;
    return right;
}

inline TreeNode *rightRotate(TreeNode *root) {
    TreeNode *left = root->left;
    root->left = left->right;
    left->right = root;
    return left;
}

inline TreeNode *leftRightRotate(TreeNode *root) {
    root->left = leftRotate(root->left);
    return rightRotate(root);
}

inline TreeNode *rightLeftRotate(TreeNode *root) {
    root->right = rightRotate(root->right);
    return leftRotate(root);
}


inline int getHeight(TreeNode *root) {
    if (root == nullptr) {
        return 0;
    }
    return max(getHeight(root->left), getHeight(root->right)) + 1;
}

inline TreeNode *insert(TreeNode *root, int &val) {
    if (root == nullptr) {
        root = new TreeNode{val};
    } else {
        if (val > root->val) {
            root->right = insert(root->right, val);
            if (abs(getHeight(root->right) - getHeight(root->left)) == 2) {
                if (val > root->right->val) {
                    root = leftRotate(root);
                } else {
                    root = rightLeftRotate(root);
                }
            }
        } else if (val < root->val) {
            root->left = insert(root->left, val);
            if (abs(getHeight(root->right) - getHeight(root->left)) == 2) {
                if (val < root->left->val) {
                    root = rightRotate(root);
                } else {
                    root = leftRightRotate(root);
                }
            }
        }
    }
    return root;
}

int main() {
    int n, x;
    cin >> n;
    TreeNode *root = nullptr;
    for (int i = 0; i < n; i++) {
        cin >> x;
        root = insert(root, x);
    }
    printf("%d", root->val);
    return 0;
}
// 1.插入点位于x的左孩子的左子树中。    左左LL     右旋。
// 2.插入点位于x的左孩子的右子树中。    左右LR    较低的先左旋，转换为LL问题，再右旋。
// 3.插入点位于x的右孩子的左子树中。    右左RL    较低的先右旋，转化为RR问题。再左旋。
// 4.插入点威武x的右孩子的右子树中。    右右RR    左旋。
```

```c++
#include "bits/stdc++.h"

using namespace std;

struct TreeNode {
    int val;
    int height;
    TreeNode *right;
    TreeNode *left;
};

inline int getHeight(TreeNode *root) {
    if (root == nullptr) {
        return 0;
    }
    return root->height;
}

inline void updateHeight(TreeNode *root) {
    root->height = max(getHeight(root->left), getHeight(root->right)) + 1;
}

inline TreeNode *leftRotate(TreeNode *root) {
    TreeNode *right = root->right;
    root->right = right->left;
    right->left = root;

    updateHeight(root);
    updateHeight(right);

    return right;
}

inline TreeNode *rightRotate(TreeNode *root) {
    TreeNode *left = root->left;
    root->left = left->right;
    left->right = root;

    updateHeight(root);
    updateHeight(left);

    return left;
}

inline TreeNode *leftRightRotate(TreeNode *root) {
    root->left = leftRotate(root->left);
    return rightRotate(root);
}

inline TreeNode *rightLeftRotate(TreeNode *root) {
    root->right = rightRotate(root->right);
    return leftRotate(root);
}


inline TreeNode *insert(TreeNode *root, int &val) {
    if (root == nullptr) {
        root = new TreeNode{val, 1};
    } else {
        if (val > root->val) {
            root->right = insert(root->right, val);
            updateHeight(root);
            if (getHeight(root->right) - getHeight(root->left) == 2) {
                if (val > root->right->val) {
                    root = leftRotate(root);
                } else {
                    root = rightLeftRotate(root);
                }
            }
        } else if (val < root->val) {
            root->left = insert(root->left, val);
            updateHeight(root);
            if (getHeight(root->left) - getHeight(root->right) == 2) {
                if (val < root->left->val) {
                    root = rightRotate(root);
                } else {
                    root = leftRightRotate(root);
                }
            }
        }
    }
    return root;
}

int n, a, b;

inline void getPath(TreeNode *root) {
    cout << root->val;
    while (root->val != b) {
        if (root->val > b) {
            root = root->left;
        } else {
            root = root->right;
        }
        cout << " " << root->val;
    }
    cout << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);
    cin >> n;
    TreeNode *root = nullptr;
    while (n--) {
        cin >> a >> b;
        if (a == 1) {
            root = insert(root, b);
        } else {
            getPath(root);
        }
    }
    return 0;
}
```

并查集

```c++
#include "bits/stdc++.h"

using namespace std;
int n, m, z, x, y;
vector<int> par;

int find(int a) {
    return par[a] == a ? a : par[a] = find(par[a]);
}

void unionn(int a, int b) {
    par[find(a)] = par[find(b)];
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0), cout.tie(0);
    cin >> n;
    par.resize(n + 1);
    for (int i = 1; i <= n; ++i) {
        par[i] = i;
    }
    return 0;
}
```

**7-3 搜索树判断**

```c++
#include "bits/stdc++.h"

using namespace std;

int tree[1010], n;
bool flag = true, flag1 = true, flag2 = true;
struct TreeNode {
    int val;
    TreeNode *right;
    TreeNode *left;
};

inline TreeNode *search(int i, int k) {
    if (i > k) {
        return nullptr;
    }
    TreeNode *root = new TreeNode{tree[i]};
    int j = i + 1;
    while (j <= k && tree[j] < tree[i]) {
        j++;
    }
    for (int l = j + 1; l < n; ++l) {
        if (tree[l] < tree[i]) {
            flag1 = false;
            return nullptr;
        }
    }
    root->left = search(i + 1, j - 1);
    root->right = search(j, k);
    return root;
}

inline TreeNode *mirror(int i, int k) {
    if (i > k) {
        return nullptr;
    }
    TreeNode *root = new TreeNode{tree[i]};
    int j = i + 1;
    while (j <= k && tree[j] >= tree[i]) {
        j++;
    }
    for (int l = j + 1; l < n; ++l) {
        if (tree[l] >= tree[i]) {
            flag2 = false;
            return nullptr;
        }
    }
    root->left = mirror(i + 1, j - 1);
    root->right = mirror(j, k);
    return root;
}

inline void print(TreeNode *root) {
    if (root == nullptr) {
        return;
    }
    print(root->left);
    print(root->right);
    if (flag) {
        flag = false;
    } else {
        cout << " ";
    }
    cout << root->val;
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++i) {
        cin >> tree[i];
    }
    TreeNode *root1 = search(0, n - 1);
    if (flag1 && root1) {
        cout << "YES\n";
        print(root1);
        return 0;
    }
    TreeNode *root2 = mirror(0, n - 1);
    if (flag2 && root2) {
        cout << "YES\n";
        print(root2);
        return 0;
    }
    cout << "NO";
}
```

