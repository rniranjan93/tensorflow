__kernel void instance_norm_meanvari_F32(
    __read_only image2d_array_t   input,
    __write_only image2d_t  output,
    float eps,
    int rsFlg,
    int input_zp,
    float input_scale,
    float input_fl,
    int width,
    int height
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, gidz, 0);
    float4 data;
    float sum = 0, sqr = 0;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    if(gidx < width)
    {
        for(coord.y = 0; coord.y < height;)
        {
            data = read_imagef(input, coord);
            coord.y++;
            sum += data.x;
            sqr += data.x * data.x;
        }
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int4 coord_out = (int4)(get_group_id(0) << 2, gidz, 0, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 dst = (float4)(0);
        dst.x = sum;
        write_imagef(output, coord_out.xy, dst);
        coord_out.x++;
        dst.x = sqr;
        write_imagef(output, coord_out.xy, dst);
    }
}

__kernel void instance_norm_meanvari_F32_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    float eps,
    int rsFlg,
    int input_zp,
    float input_scale,
    float input_fl,
    int width,
    int height
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);
    int gidy = gidz * height;

    int2 coord = (int2)(gidx, gidy);
    float4 data;
    float sum = 0, sqr = 0;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    int endH = gidy + height;
    if(gidx < width)
    {
        for(; coord.y < endH;)
        {
            data = read_imagef(input, coord);
            coord.y++;
            sum += data.x;
            sqr += data.x * data.x;
        }
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int4 coord_out = (int4)(get_group_id(0) << 2, gidz, 0, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 dst = (float4)(0);
        dst.x = sum;
        write_imagef(output, coord_out.xy, dst);
        coord_out.x++;
        dst.x = sqr;
        write_imagef(output, coord_out.xy, dst);
    }
}

__kernel void instance_norm_F32toF32(
    __read_only image2d_array_t   input,
    __read_only image2d_t   bias,
    __read_only image2d_t   scale,
    __read_only image2d_t   meanVari,
    __write_only image2d_array_t  output,
    float eps,
    int rsFlg,
    int input_zp,
    float input_scale,
    float input_fl,
    int output_zp,
    float output_scale,
    float output_fl,
    int width,
    int height,
    float dim_ratio,
    int group_num
    )
{
    int gidz = get_global_id(1);
    int4 coord = (int4)(get_global_id(0), 0, gidz, 0);
    int4 coord_para = (int4)(0, gidz, 0, 0);

    float4 gamma = read_imagef(scale, coord_para.yx);
    float4 beta  = read_imagef(bias, coord_para.yx);
    float4 mean_vari = (float4)(0);
    float scale_vari, bias_val;

    for(int i = 0; i < group_num; i++)
    {
        mean_vari.x += read_imagef(meanVari, coord_para.xy).x;
        coord_para.x++;
        mean_vari.y += read_imagef(meanVari, coord_para.xy).x;
        coord_para.x+=3;
    }
    mean_vari *= dim_ratio;
    mean_vari.s1 = mean_vari.s1 - mean_vari.s0 * mean_vari.s0 + eps;
    mean_vari.s1 = rsqrt(mean_vari.s1);

    scale_vari = gamma.s0 * mean_vari.s1;
    bias_val = (beta.s0 - scale_vari * mean_vari.s0);

    float4 data, dst;
    for(coord.y = 0; coord.y < height;coord.y++)
    {
        data = read_imagef(input, coord);

        dst.x = data.x * scale_vari + bias_val;
        write_imagef(output, coord, dst);
    }
}

__kernel void instance_norm_F32toF32_2D(
    __read_only image2d_t   input,
    __read_only image2d_t   bias,
    __read_only image2d_t   scale,
    __read_only image2d_t   meanVari,
    __write_only image2d_t  output,
    float eps,
    int rsFlg,
    int input_zp,
    float input_scale,
    float input_fl,
    int output_zp,
    float output_scale,
    float output_fl,
    int width,
    int height,
    float dim_ratio,
    int group_num
    )
{
    int gidz = get_global_id(1);
    int gidy = gidz * height;
    int2 coord = (int2)(get_global_id(0), gidy);
    int2 coord_para = (int2)(0, gidz);
    int endH = gidy + height;

    float4 gamma = read_imagef(scale, coord_para.yx);
    float4 beta  = read_imagef(bias, coord_para.yx);
    float4 mean_vari = (float4)(0);
    float scale_vari, bias_val;

    for(int i = 0; i < group_num; i++)
    {
        mean_vari.x += read_imagef(meanVari, coord_para.xy).x;
        coord_para.x++;
        mean_vari.y += read_imagef(meanVari, coord_para.xy).x;
        coord_para.x+=3;
    }
    mean_vari *= dim_ratio;
    mean_vari.s1 = mean_vari.s1 - mean_vari.s0 * mean_vari.s0 + eps;
    mean_vari.s1 = rsqrt(mean_vari.s1);

    scale_vari = gamma.s0 * mean_vari.s1;
    bias_val = beta.s0 - scale_vari * mean_vari.s0;

    float4 data, dst;
    for(; coord.y < endH; coord.y++)
    {
        data = read_imagef(input, coord);

        dst.x = data.x * scale_vari + bias_val;
        write_imagef(output, coord, dst);
    }
}
