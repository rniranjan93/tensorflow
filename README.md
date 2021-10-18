#include<bits/stdc++.h>
using namespace std;/*
#define MOD 1000000007
#define M 200005
pair<int, int> indexes[2002];
int arr[2002][2002];
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	/*int n, k;
	cin >> n >> k;
	int sum = 0;
	for (int i = 0; i < 2002; i++)
		indexes[i] = { -1,-1 };
	char c;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> c;
			if (c == 'B')
			{
				if (indexes[i].first == -1)
				{
					indexes[i].first = j;
					indexes[i].second = j;
				}
				else
					indexes[i].second = j;
			}
		}
		if (indexes[i].first == -1)
			sum++;
	}
	int maxx = 0;
	for (int i = 0; i <= n - k; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (indexes[j].first >= i && indexes[j].second <= i + k - 1)
				arr[j][i] = 1;
		}
	}
	int temp;
	for (int i = 0; i <= n - k; i++)
	{
		temp = 0;
		for (int j = 0; j < k; j++)
		{
			temp += arr[j][i];
		}
		maxx = max(maxx, temp);
		int prev = 0;
		for (int j = k; j < n; j++)
		{
			temp += arr[j][i] - arr[j-k][i];
			maxx = max(maxx, temp);
		}
	}
	cout << sum + maxx;
	return 0;
}*/
