
# Word Stand For

***ML***: Machine Learning
***MLs***: Machine Learning Systems
***MLp***: Machine Learning platforms
***NN***: Neural Network
***diff***: different | difference
***ur***: your | our
***dev***: develop | -ing | -ment
***dep***: deploy | -ing | -ment
***eva***: evaluate| -ing
***reg***: regression
***cls***: classification
***sys***: system
***swe***: software engineering

# Preface

#MachineLearning 
#Algorithm 
#Metrics 

Machine Learning systems are both complex and unique
1. **Complex**: they consist of many different components (ML algorithms, data, business logic, evaluation metrics, underlying infrastructure, etc.) and involve many different stakeholders (DS, ML Engineers, business leaders, users, even society at large)
2. **Unique**: they are data dependent, with data varying wildly from one use case to the next

In this book, we will design ML (Machine Learning) systems that are reliable, scalable, maintainable, and adaptive 

We will tackle scenarios such as: 
- Engineering data and choosing the right metrics to solve business problem
- Automating the process of continually developing, evaluating, deploying and updating models
- Developing a monitoring systems to quickly detect and address issues your models might encounter in production
- Architecting an ML platform 
- Developing responsible ML systems


## Navigating This Book

#Deployment

**The first two chapters** lay down the groundwork to set an ML project up for success. It also covers choosing the objectives for your project and how to frame your problem in a way that makes for simpler solutions

**Chapter 4 to 6** cover the pre-deployment phase of an ML project: from creating the training data and engineering features to developing and evaluating our models. 

**Chapter 7 to 9** cover the deployment and post-deployment phase. The deployed model will need to be monitored and continually updated

**Chapter 3 and 10** focus on the infrastructure needed to enable stakeholders from diff backgrounds to work together to deliver successful ML systems. **Chapter 3** focuses on data systems, whereas **Chapter 10** focuses on compute infrastructure and ML platforms. 


# Chapter 1: Overview of Machine Learning Systems

#MLOps
#DevOps

One of the first success stories of deep artificial NN in production, in November 2016, Google announced that it had incorporated its multilingual (đa ngôn ngữ) neural machine translation system into Google Translate.

Many people, when they hear "MLs", think of just the ML algorithms being used such as Logistic Reg or diff types of NN. However, the algorithms is only a small part of an MLs in production. 

![[Pasted image 20240713152027.png]]

#### Note: The Relationship Between MLOps and MLs Design

Ops in MLOps comes from DevOps, short for Developments and Operations. To operationalize something means to bring it into production, which includes dep, monitoring, and maintaining it. MLOps is a set of tools and best practices for bringing ML into production. 

MLs design takes a system approach to MLOps, which means that it considers an MLs holistically to ensure that all components and their stakeholders can work together to satisfy the specified objectives and requirements

## **1.1| When to Use ML**

**Definition**: Machine Learning is an approach to **learn complex patterns** from **existing data** and use these patterns to **make predictions on unseen data**

ML solutions will especially shine if ur problem has these following characteristics:

**1. It's repetitive**
	Most ML algorithms still require many examples to learn a pattern, not like few-shot learning for humans. When a task is repetitive, each pattern is repeated multiple times, which makes it easier for machines to learn it
	
**2. The cost of wrong predictions is cheap**
	Unless ur ML model's performance is 100% all the times, which is highly unlikely for any meaningful tasks, ur model is going to make mistakes. For example, in recommender systems, a bad recommendation is usually forgiving - the user just won't click on this. However, dev self-driving cars is challenging because an algorithmic mistake can lead to death. 
	
**3. It's at scale**
	"At scale" means diff things for diff tasks, but, in general, it means making lots of predictions. 
	Having a problem at scale means that there's lots of data for us to collect, which is useful for training ML models
	
**4. The patterns are constantly changing**
	Cultures change. Tastes change. Technologies change. What’s trendy today might be old news tomorrow.
	Because ML learns from data, you can update your ML model with new data without having figure out how the data has changed

Even if ML can't solve ur problem, it might be possible to break ur problem into smaller components, and use ML to solve some of them. For example, if ur can't build a chatbot to answer all ur customers' queries, it might be possible to build an ML model to predict whether a query matches one of the frequently asked questions. If yes, direct the customer to the answer. If not, direct them to customer service.


## **1.2| Understanding MLs**

In this section, we will go over how MLs are diff from both ML in research and traditional software
### **1.2.1| Research vs. Production**

|                            | Research                                                 | Production                               |
| :------------------------- | :------------------------------------------------------- | :--------------------------------------- |
| **Requirements**           | State-of-the-art model performance on benchmark datasets | Diff stakeholders have diff requirements |
| **Computational priority** | Fast training, high throughput                           | Fast inference, low latency              |
| **Data**                   | Static                                                   | Constantly shifting                      |
| **Fairness**               | Often not a focus                                        | Must be considered                       |
| **Interpretability**       | Often not a focus                                        | Must be considered                       |

#### **Diff stakeholders and requirements**

"Recommending the restaurants that users are most likely to click on" and "Recommending the restaurants that will bring in the most money for the app" are two diff objectives for the recommending restaurants mobile app. 

>***We'll dev one model for each objective and combine their predictions.***

Let's imagine for now that we have two diff models. Model A is the model that recommends the restaurants that users are most likely to click on, and model B is the model that recommends the restaurants that will bring in the most money for the application.  

Production having diff requirements (return restaurant recommendations in less than 100 milliseconds) from research is one of the reasons why successful research projects might not always be used in production. For example, ensembling is a technique popular for combining multiple learning algorithms to obtain better predictive performance. However, ensembling tends to make a system too complex to be useful in production, slower to make predictions or harder to interpret the results. 

#### **Computational priorities**

When designing an ML system, people who haven’t deployed an ML system often make the mistake of focusing too much on the model development part and not enough on the model deployment and maintenance part.

During model dev, training is the bottleneck. Once the model has been dep, however, its job is to generate predictions, so inference is the bottleneck. Research usually prioritizes fast training, whereas production usually prioritizes fast inference.

One corollary (hệ quả tất yếu) of this is that research prioritizes high throughput (how many queries are processed within a specific period of time) whereas production prioritizes low latency (the time it takes from receiving a query to returning the result). If ur sys always processes one query at a time, higher latency means lower throughput. 

However, because most modern distributed sys batch queries to process them together, often concurrently, *higher latency might also mean higher throughput*. If u process 10 queries at a time and it takes 10 ms to run a batch, the average latency is still 10 ms but the throughput is now 10 times higher - 1,000 queries/second. If you process 50 queries at a time and it takes 20 ms to run a batch, the average latency now is 20 ms and the throughput is 2,500 queries/second. Both latency and throughput have increased!

To reduce latency in production, you might have to reduce the number of queries you can process on the same hardware at a time.

When thinking about latency, it's important to keep in mind that latency is not an individual number but a distribution. Imagine u have 10 requests whose latencies are 100 ms, 102 ms, 100 ms, 100 ms, 99 ms, 104 ms, 110 ms, 90 ms, 3,000 ms, 95 ms. The average latency is 390 ms, which makes ur sys seem slower that actually is. 

It's usually better to think in percentiles. The most common percentile is the 50th percentile. The median of 10 request above is 100 ms, half of the requests take longer than 100 ms, and half of the requests take less than 100 ms. Higher percentiles also help u discover outliers. The 90th percentile is 3,000 ms, which is an outlier. 

Higher percentiles are important to look at because even though they account for a small percentage of ur users, sometimes they can be the most important users. Ex: on the Amazon website, the customers with the slowest requests are often those who have the most data on the accounts because they have made many purchases - that is, they are the most valuable customers.

It's a common practice to use high percentiles to specify the performance requirements for ur sys

#### **Data**

During the research phase, the datasets we work with are often clean and well-formatted, freeing you to focus on dev models. They are static

In production, data, if available, is a lot more messy. It's noisy, possibly unstructured, constantly shifting. It's likely biased, and you likely don't know how it's biased. Labels, if there are any, might be sparse, imbalanced, or incorrect. If you work with users' data, you'll also have to worry about privacy and regulatory concerns. 

#### **Fairness**

During the research phase, a model is not yet used on people, so it's easy for researchers to put off fairness as an afterthought: "Let's try to get state of the art first and worry about fairness when we get to production". When it gets to production, it's too late. 

You or someone in ur life might already be a victim of biased mathematical algorithms without knowing it. Your loan application might be rejected because the ML algorithm picks on ur zip code, which embodies biases about one's socioeconomic background, etc. Other examples of ML biases in the real world are in predictive policing algorithms, personality tests administered by potential employers, and college rankings. 

#### **Interpretability (khả năng diễn giải)**

> *"Suppose you have cancer and you have to choose between a black box AI surgeon that cannot explain how it works but has a 90% cure rate and a human surgeon with an 80% cure rate. Do you want the AI surgeon to be illegal?"*

First, interpretability is important for users, both business leaders and end users, to understand why a decision is made so that they can trust a model and detect potential biases mentioned previously. 

Second, it's important for developers to be able to debug and improve a model


### **1.2.2| MLs vs. Traditional Software**

In traditional SWE, you only need to focus testing and versioning ur code. With ML, we have to test and version our data too, and that's the hard part. How to version large dataset? How to know if a data sample is good or bad for ur sys? Not all data samples are equal - some are more valuable to ur model than others. 

The size of ML models is another challenge. As of 2022, it's common for ML models to have hundreds of millions, if not billions, of parameters, which requires GB of RAM to load them into memory. 

For now, getting these large models into production, especially on edge devices is massive engineering challenge. Then there is the question of how to get these models to run fast enough to be useful. Ex: an autocompletion model is useless if the time it takes to suggest the next character is longer than the time it takes for you to type. 


# Chapter 2: Introduction to Machine Learning Systems Design

## **1. Business and ML Objectives**

When working on a ML project, DS tend to care about the ML objectives: the metrics thay can measure about the performance of their ML models such as accuracy, F1-Score, inference latency, etc. 

But the truth is: most companies don't care about the fancy ML metrics. The DS become too focused on hacking ML metrics without paying attention to business metrics. Their manager, however, only care about business metrics. 

The ultimate goal of any project within a business is to increase profits, either directly or indirectly: 
- Directly: increasing sales and cutting costs
- Indirectly: Higher customer satisfaction and increasing time spent on a website

Many companies create their own metrics to map business metrics to ML metrics. Ex: Netflix measures the performance of their recommender sys using *take_rate* - the number of quality plays divided by the number of recommendations a user sees. The higher the take-rate, the better the recommender sys.

*Netflix also put a recommender system's take-rate in the context of their other business metrics like total streaming hours and subscription cancellation rate. They found that a higher take-rate also results in higher total streaming hours and lower subscription cancellation rates. 

The effect of an ML project on business objectives can be hard to reason about. For examples, an ML model that gives customers more personalized solutions can make them happier, which makes them spend more money on your services. The same ML model can also solve their problems faster, which makes them spend less money on ur services.

To gain a definite answer on the question of how ML metrics influence business metrics, experiments are often needed. Many companies do that with experiments like A\B testing and choose the model that leads to better business metrics, regardless of whether this model has better ML metrics


## **2. Requirements for MLs**

### **2.1. Reliability**

The sys should continue to perform the correct function at the desired level of performance even in the face of adversity (hardware or software faults, and even human error)

"Correctness" might be difficult to determine for ML systems. How do we know if a prediction is wrong if we don't have ground truth labels to compare it with?

End users don't even know that the sys has failed and might have kept on using it as if it were working. For example, if Google Translate return a sentence into a language you don't know, it might be very hard for you to tell even if the translation is wrong. 

## **2.2. Scalability**

There are multiple ways an MLs can grom. It can grow in complexity. Last year, you used a Logistic Reg model that fit into an AWS free tier instance with 1 GB of RAM, but this year, you switched to a 100-million-parameter NN that requires 16 GB of RAM to generate predictions.

Your MLs can grom in traffic volume. When you started deploying an MLs, you only served 10,000 prediction requests daily. However, as your company's user base grows, the number of prediction requests your MLs serves daily fluctuates between 1 million to 10 million.

An MLs might grow in ML model count. This growth pattern is especially common in MLs that target enterprise use cases. Initially, a startup might serve only one enterprise customer, which means this startup only has one model. However, as this startup gains more customers, they might have one model for each customers. 

When talking about scalability most people think of resource scaling, which consists of up-scaling (expanding the resources to handle growth) and down-scaling (reducing the resources when not needed).

Handling growth isn't just resource scaling, but also artifact management. Managing one hundred models is very diff from managing one model. With one model, you can, perhaps, manually monitor this model's performance and manually update the model with new data. However, with one hundred models, both the monitoring and retraining aspect will need to be automated. 

## **2.3. Maintainability**

There are many people who will work on an MLs. They are ML engineers, DevOps engineers, and subject matter experts (SMEs). They might come from very diff backgrounds, with very diff programming languages and tools, and might own diff parts of the process. 

It's important to structure your workloads and set up your infrastructure. Code should be documented. Code, data and artifacts should be versioned. Models should be sufficiently reproducible so that even when the original authors are not around, other contributors can have sufficient contexts to build on their work. 


## **2.4. Adaptability**

To adapt to shifting data distributions and business requirements, the sys should have some capacity for both discovering aspects for performance improvement and allowing updates without service interuption.


# **3. Iterative Process**

Developing an MLs is an iterative and, in most cases, never-ending process. One a sys is put into production, it'll need to be continually monitored and updated.

Before deploying my first MLs, I thought the process would be linear and straightforward. I thought all I had to do was to collect data, train a model, deploy that model, and be done. However, I soon realized the process looks more like a cycle with a lot of back and forth between diff steps.

For example, here is one workflow that you might encounter when building an Ml model to predict whether an ad should be shown when users enter a search query:

1. Choose a metric to optimize. For example, you might want to optimize for impressions - the number of times an ad is shown
2. Collect data and obtain labels.
3. Engineer features.
4. Train models.
5. During error analysis, you realize that errors are caused by the wrong labels, so you relabel the data.
6. Train the model again
7. During error analysis, you realize that model always predicts that an ad shouldn't be shown, and the reason is because 99.99% of the data you have NEGATIVE labels. So you have to collect more data of ads that should be shown.
8. Train model again
9. The model performs well on ur existing test data, which is by now two months old. However, it performs poorly on the data from yesterday. Ur model is now state, so you need to update it on more recent data. 
10. Train the model again.
11. Deploy the model.
12. The model seems to be performing well, but then the business people come knocking on ur door asking why the revenue is decreasing. It turns out the ads are being shown, but few people click on them. So u wanna change ur model to optimize for ad click-through rate instead. 
13. Go to step 1


![[Pasted image 20240714170915.png]]

*Step 1: Project scoping*
	A project starts with scoping the project, laying out goals, objectives, and constraints. Stakeholders should be identified and involved. Resources estimated and allocated. 
	
*Step 2: Data engineering*
	Developing ML models starts with engineering data, which covers handling data from diff sources and formats.
	
*Step 3: ML model dev*
	Consisting of extracting features and developing initial models leveraging these features
	
*Step 4: Deployment*
	After a model is developed, it needs to be made accessible to users. 
	
*Step 5: Monitoring and continual learning*
	Once in production, models need to be monitored and maintained to be adaptive to changing environments and changing requirements.
	
*Step 6: Business analysis*
	Model performance needs to be evaluated against business goals and analyzed to generate business insights. This step is closely related to the first step

# **4. Framing ML Problems**

In this section, we'll focus on two aspects: the output of your model and the objective function that guides the learning process

## **4.1. Types of ML Tasks**

The most general types of ML tasks are cls and reg. 

### **Cls vs. Reg**

...

### **Multiclass Cls**

When the number of classes is high, such as disease diagnosis where the number of diseases can go up to thousands or product classification where the number of products can go up to tens of thousands, we say the cls task has *high cardinality*

High cardinality problems can be very challenging. The first challenge is in data collection. ML models typically need at least X examples for each class to learn to classify that class. So if you have 1,000 classes, you already need at least 1000X examples. In author's experience, X is at least 100.  The data collection can be especially difficult for rare classes.

When the number of classes is large, hierarchical classification might be useful. In hierarchical cls, you have a classifier to first classify each example into one of the large groups. Then you have another classifier to classify this example into one of the subgroups. For examples, for product classification, you can first classify each product into one of the four main categories: electronics, home & kitchen, fashion or pet supplies. After a product has been classified into fashion class, you can use another classifier to put this product into one of the subgroups like shoes, shirts, etc.

### **Multiclass vs. Multilabel**

In both binary and multiclass classification, each example belongs to exactly one class. When an example can belong to multiple classes, we have a *multilabel classification* problem. For example, when we build a model to classify articles into four topics - tech, entertainment, finance, and politics - an article can be in both tech and finance.

There are two major approaches to multilabel cls problems. 

The first is to treat it as you would a multiclass cls. In multiclass cls, if there are four possible classes [tech, entertainment, finance, politics] and the label for an example is entertainment, you represent this label with the vector `[0, 1, 0, 0]`. In multilabel cls, if an example has both labels entertainment and finance, its label will be represented as `[0, 1, 1, 0]`

The second approach is to turn it into a set of binary cls problems. For the article cls problem, you can have four models corresponding to four topics, each model outputting whether an article is in that topic or not.


## **4.2. Objective Functions**

To learn, an ML model needs an objective function to guide the learning process. 
An objective function is also called a loss function, because the objective of the learning process is usually to minimize (or optimize) the loss caused by wrong predictions. 

### **Decoupling objectives**

If you want optimize both *quality_loss* - the difference between each's post predicted quality and its true quality and *engagement_loss* - the difference between each's post predicted clicks and its actual number of clicks, one approach is to combine them into one loss and train one model to minimize that loss:

$$loss = \alpha \times quality\_loss + \beta \times engagement\_loss$$
A problem with this approach is that each time you tune $\alpha$ and $\beta$ - for example, if the quality of newsfeeds goes up but users' engagement goes down, you might want to decrease $\alpha$ and increase $\beta$ - you'll have to retrain your model

Another approach is train two different models, each optimizing one loss. So you have two models: *quality_model & engagement_model*

You can combine the models' output and rank posts by their combined score:

$$\alpha \times quality\_score + \beta \times engagement\_score$$
Now you can tweak $\alpha$ and $\beta$ without retraining your models!

In general, when there are multiple objectives, it's a good idea to decouple them first because it makes model development and maintenance easier


# **5. Mind Versus Data**

Progress in the last decade shows that the success of an MLs depends largely on the data it was trained on. Instead of focusing on improving ML algorithms, most companies focus on managing and improving the data.

In this section, *Mind* might be intelligent architectural designs or algorithms. 

In the mind-over-data camp, "ML will not be the same in 3-5 years, and ML forks who continue to follow the current data-centric paradigm will find themselves outdated"

Many people in ML today are in the data-over-mind camp. If you want to use data science, and improve your products or processes, you need to start with building out your data, both in terms of quality and quantity. 

No one can deny that data is essential, for now. Both the research and industry trends in the recent decades show the success of ML relies more and more on the quality and quantity of data. Models are getting bigger and using more data

# Chapter