#include<bits/stdc++.h>
using namespace std;
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int t;
	long long n, l, r,index,a;
	cin >> t;
	while (t--)
	{
		cin >> n >> l >> r;
		a = n*(n-1)+1 - l + 1;
		if (a == 1)
		{
			cout << "1\n";
			continue;
		}
		else
		{
			long long start = 1, end = n-1;
			while (start <= end)
			{
				long long mid = (start + end) / 2;
				if (a > mid * (mid + 1) + 1)
				{
					start = mid+1;
				}
				else
				{
					end = mid - 1;
				}
			}
			index =start;
		}
		int i = n - index;
		int j = i + 1 + ((index * (index + 1) + 1) - a) / 2;
		while (l <= r)
		{
			if (i == n)
			{
				cout << "1 ";
				break;
			}
			if (l % 2)
			{
				cout << i << ' ';
			}
			else
			{
				cout << j << ' ';
				j++;
			}
			if (j > n)
			{
				i++;
				j = i + 1;
			}
			l++;
		}
		cout << endl;
	}
	return 0;
}
