#include<bits/stdc++.h>
using namespace std;
int main()
{
	/*ios_base::sync_with_stdio(false);
	cin.tie(NULL);*/
	int n, a;
	int count = 0;
	int left = 0;
	cin >> n;
	cin >> a;
	count += a / 3;
	left = a % 3;
	for (int i = 1; i < n; i++)
	{
		cin >> a;
		if (a % 2)
		{
			if (a == 1)
			{
				left++;
			}
			else
			{
				a -= 3;
				count++;
				if (2 * left < a)
				{
					a -= 2 * left;
					count += left + a / 3;
					left = a % 3;
				}
				else
				{
					count += a / 2;
					left -= a / 2;
				}
			}
		}
		else
		{
			if (2 * left < a)
			{
				a -= 2 * left;
				count += left + a / 3;
				left = a % 3;
			}
			else
			{
				count += a / 2;
				left -= a / 2;
			}
		}
	}
	cout << count;
	return 0;
}
