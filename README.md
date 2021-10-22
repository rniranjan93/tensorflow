#include<bits/stdc++.h>
using namespace std;
#pragma warning(disable:4996)
vector<int>v;
vector<vector<long long>>vv(3003, vector<long long>(3003));
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int n;
	long long a;
	cin >> n;
	int cnt;
	v = vector<int>(n);
	for (int i = 0; i < n; i++)
	{
		cin >> a;
		cnt = 0;
		while (a)
		{
			if (a & 1)
				cnt++;
			a = a >> 1;
		}
		v[i] = cnt;
	}
	long long sum = 0;
	for (int i = n-1; i >= 0; i--)
	{
		for (int j = 3002; j >= 0; j--)
		{
			if (vv[j][i + 1] == 0)
				continue;
			for (int k = j + v[i]; k >= max(v[i], j) - min(v[i], j); k -= 2)
			{
				if(k<=3002)
					vv[k][i] = max(vv[k][i], vv[j][i + 1]);
			}
		}
		vv[v[i]][i]++;
		sum += vv[0][i];
	}
	cout << sum;
	return 0;
}
