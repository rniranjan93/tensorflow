#include<bits/stdc++.h>
using namespace std;
#pragma warning(disable:4996)
vector<int>v(200005);
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int t;
	int n;
	cin >> t;
	while (t--)
	{
		cin >> n;
		for (int i = 0; i < n; i++)
			cin >> v[i];
		sort(v.begin(), v.begin() + n);
		pair<int, int>p = { -1,-1 };
		set<pair<int, int>>s;
		for (int i = 0; i < n; i++)
		{
			if (p.second != v[i])
			{
				if (p.second != -1)
				{
					s.insert(p);
				}
				p.second = v[i];
				p.first = 0;
			}
			p.second++;
		}
		s.insert(p);

	}
	return 0;
}
