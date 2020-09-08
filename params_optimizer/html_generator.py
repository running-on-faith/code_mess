"""
@author  : MG
@Time    : 2020/9/8 9:55
@File    : html_generator.py
@contact : mmmaaaggg@163.com
@desc    : 用于
"""
import os


def generate_html(file_path, labels, x_label, y_label, z_label, color_label, symbol_size_label):
    content_labels = '[\n' + \
                     ',\n'.join([f"\t\t\t{{"
                                 f"\n\t\t\t\tname: '{_}', "
                                 f"\n\t\t\t\tindex: {n}\n\t\t\t}}" for n, _ in enumerate(labels)]) + \
                     '\n\t\t\t];'
    content = """<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <title>ECharts</title>
    <script src="./Js/jquery-3.5.1.js"></script>
    <script src="https://cdn.bootcss.com/echarts/4.2.1-rc1/echarts.min.js"></script>
    <script src="./Js//echarts-gl.js"></script>
    <!-- <script src="lib/echarts-gl.min.js"></script> -->
    <style>
        html,
        body,
        #app {
            width: 100%;
            height: 100%;
            /* background: rgba(0, 0, 0, .5) */
        }
    </style>
</head>

<body>
    <!-- 为ECharts准备一个具备大小（宽高）的Dom -->
    <div id="app"></div>
    <script src="./data.js"> </script>
    <script type="text/javascript">
        // 基于准备好的dom，初始化echarts实例
        let app = echarts.init(document.getElementById('app'))

        var indices = {
            name: 0,
            group: 1,
            id: 16
        };
        var schema = """ + content_labels + \
              """
          var data;
      
          var fieldIndices = schema.reduce(function (obj, item) {
              obj[item.name] = item.index;
              return obj;
          }, {});
      
          var groupCategories = [];
          var groupColors = [];
          var data;
          var fieldNames = schema.map(function (item) {
              return item.name;
          });
      
          // fieldNames = fieldNames.slice(2, fieldNames.length - 2);
          // console.log(fieldNames)
      
      
          function getMaxOnExtent(data) {
              var colorMax = -Infinity;
              var symbolSizeMax = -Infinity;
              var colorMin = Infinity;
              var symbolSizeMin = Infinity;
      
              for (var i = 0; i < data.length; i++) {
                  var item = data[i];
                  var colorVal = item[fieldIndices[config.color]];
                  var symbolSizeVal = item[fieldIndices[config.symbolSize]];
                  colorMax = Math.max(colorVal, colorMax);
                  colorMin = Math.min(colorVal, colorMin);
                  symbolSizeMax = Math.max(symbolSizeVal, symbolSizeMax);
                  symbolSizeMin = Math.min(symbolSizeVal, symbolSizeMin);
              }
              return {
                  max: {
                      color: colorMax,
                      symbolSize: symbolSizeMax,
                  },
                  min: {
                      color: colorMin,
                      symbolSize: symbolSizeMin,
                  }
              };
          }
      
          var config = app.config = {
              xAxis3D: '""" + x_label + """',
              yAxis3D: '""" + y_label + """',
              zAxis3D: '""" + z_label + """',
              color: '""" + color_label + """',
              symbolSize: '""" + symbol_size_label + """',
      
              onChange: function () {
                  var max = getMaxOnExtent(data);
                  if (data) {
                      app.setOption({
                          visualMap: [{
                              max: max.max.color / 2
                          }, {
                              max: max.max.symbolSize / 2
                          }],
                          xAxis3D: {
                              name: config.xAxis3D
                          },
                          yAxis3D: {
                              name: config.yAxis3D
                          },
                          zAxis3D: {
                              name: config.zAxis3D
                          },
                          series: {
                              dimensions: [
                                  config.xAxis3D,
                                  config.yAxis3D,
                                  config.yAxis3D,
                                  config.color,
                                  config.symbolSiz
                              ],
                              data: data.map(function (item, idx) {
                                  return [
                                      item[fieldIndices[config.xAxis3D]],
                                      item[fieldIndices[config.yAxis3D]],
                                      item[fieldIndices[config.zAxis3D]],
                                      item[fieldIndices[config.color]],
                                      item[fieldIndices[config.symbolSize]],
                                  ];
                              })
                          }
                      });
                  }
              }
          };
          app.configParameters = {};
      
          ['xAxis3D', 'yAxis3D', 'zAxis3D', 'color', 'symbolSize'].forEach(function (fieldName) {
              app.configParameters[fieldName] = {
                  options: fieldNames
              };
          });
      
          var max = getMaxOnExtent(data);
          app.setOption({
              tooltip: {},
              visualMap: [{
                  top: 10,
                  calculable: true,
                  dimension: 3,
                  max: max.max.color / 2,
                  min: max.min.color,
                  inRange: {
                      color: ['#1710c0', '#0b9df0', '#00fea8', '#00ff0d', '#f5f811', '#f09a09', '#fe0300']
                  },
                  textStyle: {
                      color: '#fff'
                  }
              }, {
                  bottom: 10,
                  calculable: true,
                  dimension: 4,
                  max: max.max.symbolSize / 2,
                  min: max.min.symbolSize,
                  inRange: {
                      symbolSize: [10, 40]
                  },
                  textStyle: {
                      color: '#fff'
                  }
              }],
              xAxis3D: {
                  name: config.xAxis3D,
                  type: 'value',
                  scale: true
              },
              yAxis3D: {
                  name: config.yAxis3D,
                  type: 'value',
                  scale: true
              },
              zAxis3D: {
                  name: config.zAxis3D,
                  type: 'value',
                  scale: true
              },
              grid3D: {
                  environment : 'rgba(0,0,0,.5)',
                  axisLine: {
                      show: true,
                      lineStyle: {
                          color: '#fff'
                      }
                  },
                  axisPointer: {
                      lineStyle: {
                          color: '#ffbd67'
                      }
                  },
                  viewControl: {
                      // autoRotate: true
                      // projection: 'orthographic'
                  }
              },
              series: [{
                  type: 'scatter3D',
                  dimensions: [
                      config.xAxis3D,
                      config.yAxis3D,
                      config.yAxis3D,
                      config.color,
                      config.symbolSiz
                  ],
                  data: data.map(function (item, idx) {
                      return [
                          item[fieldIndices[config.xAxis3D]],
                          item[fieldIndices[config.yAxis3D]],
                          item[fieldIndices[config.zAxis3D]],
                          item[fieldIndices[config.color]],
                          item[fieldIndices[config.symbolSize]],
                          idx
                      ];
                  }),
                  symbolSize: 12,
                  // symbol: 'triangle',
                  itemStyle: {
                      borderWidth: 1,
                      borderColor: 'rgba(255,255,255,0.8)'
                  },
                  emphasis: {
                      itemStyle: {
                          color: '#fff'
                      }
                  }
              }]
          });
      </script>
      </body>
      
      </html>
          """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path


def _test_generate_html():
    output_labels = ['short', 'long', 'signal', 'calmar', 'cagr', 'daily_sharpe', 'period']
    html_file_path = generate_html(
        os.path.join('html', 'index.html'),
        labels=output_labels,
        x_label=output_labels[0],
        y_label=output_labels[6],
        z_label=output_labels[1],
        color_label=output_labels[3],
        symbol_size_label=output_labels[2],
    )
    print(html_file_path)


if __name__ == "__main__":
    _test_generate_html()
