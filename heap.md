

**pta L2-012 关于堆的判断**

```c++
#include <bits/stdc++.h>
using namespace std;
string s;
int h[1010], n, m, t;
map<int, int> mp;

void up(int x) {
    while (x / 2 && h[x / 2] > h[x]) {
        swap(h[x / 2], h[x]);
        x /= 2;
    }
} 

int main() {
    ios::sync_with_stdio(false);
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> h[i];
        up(i);
    }
    for (int i = 1; i <= n; i++)
        mp[h[i]] = i;
    int x, y;
    while (m--) {
        cin >> x >> s;
        if (s[0] == 'a') {
            cin >> y;
            getline(cin, s);
            if (mp[x] / 2 == mp[y] / 2) cout << "T" << endl;
            else cout << "F" << endl;
        } else {
            cin >> s >> s;
            if (s[0] == 'r') {
                if (mp[x] == 1) cout << "T" << endl;
                else cout << "F" << endl;
            } else if (s[0] == 'p') {
                cin >> s >> y;
                if (mp[x] == mp[y] / 2) cout << "T" << endl;
                else cout << "F" << endl;
            } else {
                cin >> s >> y;
                if (mp[x] / 2 == mp[y]) cout << "T" << endl;
                else cout << "F" << endl;
            }
        }
    }
    return 0;
}

```
