[TOC]

<font color=#FF0000 >红色表问题</font> 

<font color=#008000 >绿色表to-do</font> 

<font color=#0000FF >蓝色表重点</font>

【】表一些解释，附注

**关于小数据集的论文是：Information Asymmetries**

# Intro

1. 本文在开头就提出来了：本文广泛分析了信息不对称的影响，以及Swoopo的feature：bidpacks和SwoopItNow选项，从而量化这些拍卖中不完善信息的影响。
   
   > While previous modeling work predicts profit-free equilibria, we analyze the impact of *information asymmetry* broadly, as well as **Swoopo features such as bidpacks and the Swoop It Now option specifically, to quantify the effects of imperfect information** in these auctions.
   
   更多的对于paper模型的介绍可以看Goodnotes上的笔记分析

2. Q：数据来源于什么网站？
   
   A：只来自于Swoopo

# Outcome数据集：

- 基于Swoopo直接发布的信息，其中包含了关于拍卖的有限信息。所提供的信息包括产品描述、零售价格、最终拍卖价格、**投标费、价格增量**等基本功能。该数据集涵盖了超过121,419次拍卖。采集于**2008年9月8日至2009年12月12日期间**。【<font color=#0000FF >对比Trace采集于2009年10月1日至2009年12月12日</font>】

## 列属性：

- price: The price **the auction reached**, in dollars。
- finalprice: The price **charged to the winner** in dollars。用户实际上付的钱是这个
  - Q：finalprice和price和bidfee有什么关系？
    
    A：finalprice为0表示winner只需要交bidfees就可以带走商品

- **bidfee**：每次bid用户的付费。数据集里只有**60和75美分**两种bidfee，前者占比0.36。
  - 这个bid fee用户是可以以折扣价购买得到的，而且并没有关于折扣的数据。本文假设所有用户有同样的bid fee

> While we know the final purchase price exactly, we can only estimate bid fees, **as some bidders have access to discounted bids**, for reasons we discuss in detail in Section 5. We therefore overestimate bid fees by **assuming all bidders pay the standard bid fee**.
> 
> when calculating the profit of a bidder, we assume a fixed bid cost of 60 cents, **as we cannot determine the true cost**. This could affect our interpretation of the results.

- **bid increment**：每次bid商品价格的增长值。有1 2 6 12 15 等

- placedbids:  The **number** of paid bids placed by the winner，和下面这个attr.是一对的

- **freebids**: The **number** of free bids place by the winner。
  
  - 注意这里应该是一个<u>抵多少“次”bid的概念</u>，而不是抵多少钱。文章对这个的解释：
    
    > Less obviously, **not all** bidders on Swoopo are paying the same price per bid, since one item available for auction at Swoopo is a **bidpack**, which is effectively **an option to make a fixed number of bids in future auctions for free (“freebids”)**. Players that win bidpacks at a discount on face value therefore have the power to make bids at a cheaper price than other players.
  
  - 也就是用户可以先经过拍卖得到bidpack（这个网站设计的虚拟商品），然后接下来make a fixed number of bids in future auctions for free (“freebids”)。同时，在数据集里有名为*50-freebids-voucher*这样的商品也可以印证这个bidpack的存在。<font color=#FF0000 >就是免费bid的50次要连续的用完？</font>【不过这个问题应该不重要 】
  - 作者在 5 Asymmetries in Bid Fees这一section的5.1 Motivation 里讲了bidpacks的影响；由于bidpack的存在，一些用户在bid上花的钱是不一样的，这个“影响”无法估计，本文的策略是：<font color=#0000FF >在这一节之外的地方，考虑其他非对称情况时，认为所有用户都花了一样的bid fee</font>
  
  > While we know the **final purchase** price exactly, we can only estimate bid fees, **as some bidders have access to discounted bids**, for reasons we discuss in detail in Section 5. We therefore **overestimate bid fees by assuming all bidders pay the standard bid fee**. On the other hand, Swoopo’s stated retail value for the item tends to be above market rate, so by using the stated retail value for our calculation we **underestimate** Swoopo’s profit. We do not suggest these effects simply cancel each other out, but we believe our estimate provides a suitable ballpark figure.

- flg_click_only: 不允许使用BidButler代理的拍卖。A binary flag indicating a "NailBiter'' auction。 
  
  - *BidButlers*机制：由Swoopo提供的自动bid代理。
  
  > “NailBiter” auctions ： Swoopo auctions which **do not permit** the use of automated bids by a “BidButler”

- flg_beginnerauction: A binary flag indicating a beginner auction

- flg_endprice: A binary flag indicating a 100%-off auction。可以看到这个flag为1的商品的finalprice都是0。

- flg_fixedprice: 买家拍卖的实际上是以固定价格买走该商品的权利. A binary flag indicating a fixed-price auction
  
  数据集里有这3个binary flag，论文里的解释：
  
  > Regular auctions exclude **“NailBiter” auctions** which do not allow the use of BidButlers, **beginner auctions** which are for players who have not won an auction previously, and **fixed-price auctions**. 
  
  - 在这个论文里**Regular auctions**下不包括以上这3种拍卖。论文Figure 2展示了Swoopo在Regular auctions上获得的月利润率；同时这一段话也展示了Swoopo更多的拍卖类型，比如fixed-price拍卖，同时也叫做100% off auction：
  
  > An important variation is a fixed-price auction, where the winner **obtains the right to buy the item at a fixed price *p*. **When *p* = 0, such an auction is referred to as a 100% off auction; in this case Swoopo derives all of its revenue from the bids.

# Trace数据集

- 采集于**2009年10月1日至2009年12月12日**之间，Trace数据集是自己记录的对实时拍卖的跟踪。包括来自Swoopo拍卖的相同信息，以及每次拍卖的详细投标信息，特别是与每次投标相关的时间和玩家。该数据集涵盖了7,353次拍卖【下文可以看到<font color=#0000FF >实际上有3025次auction的数据不全，</font>另外附录B里这个数据写的是7352，应该是笔误】和2,541,332次投标。【它实际上是按照时间间隔探测的，当离结束还有2min时，会每秒至少探测一次，下面有具体的爬取策略】，当一次探测探测到多个投标者会打上相同的标签（原文里描述的是多个元组tuple），但是网站最多只能提供10个：

>  Our methodology to collect bidding information entailed **continuous monitoring** of Swoopo auctions. We **probe** Swoopo according to a varying **probing interval** that is described in detail in Appendix B; in particular,when the auction clock is at **less than 2 minutes**, we probe **at least once a second**. Swoopo responds with **a list of up to ten tuples** of the form (*username, bidnumber*) indicating the players that placed a bid since our previous probe and the order in which they did so.

- <font color=#0000FF >只有 4,328次记录了全部的bid(no missing bids)，有3025次auction的数据不全</font>，而且在4328次中，有3026次是 NailBiter auctions。<font color=#FF0000 >下文最后那个“these”指的是什么？Missing bids？</font> 

>  Overall, we captured every bid from 4,328 auctions. The remaining 3,025 auctions had a total of 491,360 missing bids; we did not consider these in our study.
> 
>  3,026 complete (no missing bids) “NailBiter” auctions 

- 因为是自己根据时间间隔爬取的，所以为了避免不必要的网站访问，设置了一定的爬取规则：（back-off后退策略）另外当有人出价时，商品的倒计时会多+x秒。

> 初始探测间隔被设置为1秒。
> 
> 当倒计时钟还剩10分钟以上时，如果上次的爬虫请求中没有bid，就把探测间隔增加半秒，最多一分钟；如果爬到了新的bid，就返回1s探测间隔
> 
> 当还有2-10min结束拍卖时，这个间隔最大只会加到10秒钟。
> 
> <font color=#0000FF >最后2min时Swoopo会给用户建议何时出价</font>，我们取建议时间和1秒钟的最小值去爬数据

- 网站有<font color=#0000FF >Swoop IT NOW </font>特性，作者说是2009年7月，也就是采集Trace数据集之前新添加上的，它允许竞标失败的用户以（retail price - all bidding fee）的价格买走这个商品
  - 在8.1 *Swoop It Now and Chicken*被提到，进行了分析：当多个玩家考虑利用这个Swoop it NOW选项作为后备选项时，游戏就会成为chicken的变体。
  - <font color=#0000FF >但是Swoop it now使用的频率不得而知</font>。这些信息既不是由Swoopo提供的，也不是从Swoopo提供的任何数据中衍生出来的。作者怀疑这一功能经常被忽视，或者是买家对于Swoopo网站对商品过高的零售价定价不感兴趣。

## 列属性

- auction_id: the auction id. **和Outcome的auction_id是一致的**

- bid_time: the date and time of the bid

- bid_ct: bid发出时的倒计时剩余时间 the value of the countdown clock at the time the bid was reported to us, in seconds。

- bid_number: 第几次bid。the number of the bid (1 for the first bid, 2 for the second one, and so on) 

- bid_user: the username of the bidder

- bid_butler: 是否用了bid代理机器人。1 for BidButler bids, 0 otherwise

- bid_cp: bid之后商品的最新价格。 the price of the item after the bid was placed

- bid_user_secs_added: 用户手动出价给倒计时添加 了几秒。才会增加这个秒数。the number of seconds added to the countdown clock as a result **non-BidButler **bids in this bid group

- bid_butler_secs_added: 代理机器人BidButler出价给倒计时添加了几秒。 the number of seconds added to the countdown clock as a result **BidButler** bids in this bid group

- bid_infered: 是否是该商品的最后出价. 0 for the final bid in a bid group, 1 otherwise (see note below)

- bid_group: 在连续的探测之间，偶尔会有**多个出价会一起**被探测到。 我们称之为投标组**bid group**。 一个投标组中的所有投标都有相同的时间戳。bids that were reported as part of the same group will have the same group number

- bid_final: 1 for the winning (ie, last) bid, 0 otherwise
