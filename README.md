#include<bits/stdc++.h>
using namespace std;
/*
struct node
{
	int h;
	int w;
	node(int x1, int x2)
	{
		h = x1;
		w = x2;
	}
};*/
int tot = 1e7 + 2;
vector<int> v(tot, 1);
map<int, int> mp;
int main()
{
	int t;
	cin >> t;
	for (int i = 2; i < tot; i++)
	{
		for (int j = 1; i * j < tot; j++)
			v[i * j] += i;
	}
	for(int i=1;i<tot;i++)
		if (mp[v[i]] != 0)
		{
			mp[v[i]] = i;
		}
	int l;
	while (t--)
	{
		cin >> l;
		cout << mp[l] << endl;
	}
	/*int n;
	int a, b;
	vector<pair<node, int>>v;
	while (t--)
	{
		cin >> n;
		for (int i = 0; i < n; i++)
		{
			cin >> a >> b;
			v.push_back(make_pair(node(a,b),i));
			v.push_back(make_pair(node(b, a), i));
		}
		sort(v.begin(), v.end(), [](pair<node, int> &x1,pair<node, int> &x2) {node& a = x1.first;node& b = x2.first; if (a.h < b.h)return true; if (a.h == b.h && a.w < b.w)return true; return false; });
		cout << "h";
	}*/
	return 0;
}

