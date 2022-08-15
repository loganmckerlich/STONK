import helpers as hf

def beat_market(stock_name,length,tt,params,feature):

    combo = hf.make_stock_data(stock_name,'change')

    out,p,a= hf.walking_sarimax(combo.tail(length),tt,params)

    out_plot = hf.quick_plot(out)

    print(f'sarimax rmse: {hf.rmse(p,a)}') #14.16 

    all_compare,compare_plot = hf.run_test(out,feature)

    return out,out_plot,all_compare,compare_plot


if __name__ == "__main__":
    stock_name = ['pcar','amzn','tsla','oil','aapl']
    params = {'order':[5,0,0],
            'seasonal_order':[0,0,0,5],
            'trend':'ct'}
    length = 500
    tt = 0.8

    feature = 'Close'

    out,out_plot,all_compare,compare_plot = beat_market(stock_name,length,tt,params,feature)