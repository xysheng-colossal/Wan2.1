import torch
import torch_npu
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import functools

class Parallel_VAE_SP:
    def __init__(self, h_split=1, w_split=1, world_size=None):
        """
        Initialize distributed parallel processing parameters
        
        Args:
            h_split (int): Number of splits along height dimension
            w_split (int): Number of splits along width dimension
            world_size (int): Total number of processes (default: current world size)
        """
        if world_size is None:
            world_size = dist.get_world_size()  # Get total process count [[1]][[6]]
            
        # Validate world_size matches grid dimensions
        assert w_split * h_split == world_size, (
            f"world_size must be {w_split} * {h_split} = {w_split*h_split}, but got {world_size}"
        )
        
        self.rank = dist.get_rank()  # Current process rank [[6]]
        self.world_size = world_size
        self.w_split = w_split
        self.h_split = h_split
        
        # Calculate grid coordinates
        self.row_rank = self.rank // w_split  # Row index (0 to w_split-1) [[6]]
        self.col_rank = self.rank % w_split   # Column index (0 to h_split-1) [[6]]
        
        # Create communication groups
        self._create_group_by_row()
        self._create_group_by_col()
        
        self.ori_conv3d = None

    def _create_group_by_row(self):
        """Create process groups for row-wise communication"""
        for r in range(self.h_split):
            ranks_in_row = []
            for c in range(self.w_split):
                global_rank = r * self.w_split + c
                ranks_in_row.append(global_rank)
                row_group = dist.new_group(ranks=ranks_in_row)
                if r == self.row_rank:
                    self.row_group = row_group

    def _create_group_by_col(self):
        """Create process groups for column-wise communication"""
        for c in range(self.w_split):
            ranks_in_col = []
            for r in range(self.h_split):
                global_rank = r * self.w_split + c
                ranks_in_col.append(global_rank)
                col_group = dist.new_group(ranks=ranks_in_col)
                if c == self.col_rank:
                    self.col_group = col_group

    def __call__(self, x):
        """Split input tensor across last two dimensions"""
        x = x.chunk(self.w_split, dim=-1)[self.col_rank]
        x = x.chunk(self.h_split, dim=-2)[self.row_rank]
        return x

    def patch(self, x):
        """
        Partition input tensor into grid blocks and record partition shapes
        
        Args:
            x (torch.Tensor): Input tensor with shape [b, c, t, h, w]
            
        Returns:
            torch.Tensor: Local partition tensor for current process
        """
        # Get input dimensions
        batch_size, channels, time_steps, height, width = x.shape
        
        # Calculate base partition dimensions
        base_patch_height = height // self.h_split
        base_patch_width = width // self.w_split
        remainder_height = height % self.h_split
        remainder_width = width % self.w_split
        
        # Generate partitions
        patches = []
        for r in range(self.h_split):
            for c in range(self.w_split):
                # Calculate current partition dimensions
                patch_height = base_patch_height + (1 if r < remainder_height else 0)
                patch_width = base_patch_width + (1 if c < remainder_width else 0)
                
                # Calculate partition boundaries
                start_h = r * base_patch_height + min(r, remainder_height)
                end_h = start_h + patch_height
                start_w = c * base_patch_width + min(c, remainder_width)
                end_w = start_w + patch_width
                
                # Extract partition
                patch = x[:, :, :, start_h:end_h, start_w:end_w]
                patches.append(patch)
        
        # Get local partition
        local_patch = patches[self.rank]
        
        # Create rank mappings for communication
        self.row_to_global_rank = {
            r: r * self.w_split + self.col_rank 
            for r in range(self.h_split)
        }
        self.col_to_global_rank = {
            c: self.row_rank * self.w_split + c 
            for c in range(self.w_split)
        }
        
        return local_patch 

    def dispatch(self, local_patch):
        """
        Reconstruct full tensor through two-stage all-gather
        
        Args:
            local_patch (torch.Tensor): Local partition tensor
            
        Returns:
            torch.Tensor: Reconstructed full tensor
        """
        # First all-gather to collect partition shapes
        local_shape = torch.tensor(local_patch.shape[-2:], 
                                   device=local_patch.device, dtype=torch.int32)
        shape_list = [torch.empty(2, dtype=torch.int32, 
                     device=local_patch.device) for _ in range(self.world_size)]
        dist.all_gather(shape_list, local_shape)
        
        all_shapes = [tuple(shape.tolist()) for shape in shape_list]
        
        # Calculate original dimensions
        total_h = 0
        total_w = 0
        row_heights = {}  # Track row heights
        col_widths = {}   # Track column widths
        
        for rank in range(self.world_size):
            r_rank = rank // self.w_split
            c_rank = rank % self.w_split
            h_part, w_part = all_shapes[rank]
            
            # Record first occurrence of row height
            if r_rank not in row_heights:
                row_heights[r_rank] = h_part
            # Record first occurrence of column width
            if c_rank not in col_widths:
                col_widths[c_rank] = w_part
                
        total_h = sum(row_heights.values())
        total_w = sum(col_widths.values())
        
        # Prepare buffers for data gathering
        batch_size, channels, time_steps = local_patch.shape[:3]
        gathered_data = [
            torch.empty(
                (batch_size, channels, time_steps, h_part, w_part),
                device=local_patch.device,
                dtype=local_patch.dtype
            ) for h_part, w_part in all_shapes
        ]
        
        # Second all-gather to collect partition data
        dist.all_gather(gathered_data, local_patch.clone())
        
        # Reconstruct full tensor
        full_tensor = torch.empty(
            (batch_size, channels, time_steps, total_h, total_w),
            device=local_patch.device, 
            dtype=local_patch.dtype
        )
        
        current_row = 0
        for r in range(self.h_split):
            current_col = 0
            row_height = row_heights[r]
            for c in range(self.w_split):
                rank = r * self.w_split + c
                h_part, w_part = all_shapes[rank]
                
                # Place partition in correct position
                full_tensor[:, :, :, current_row:current_row+h_part, 
                            current_col:current_col+w_part] = gathered_data[rank]
                current_col += col_widths[c]
            current_row += row_height
            
        return full_tensor

    def exchange_columns(self, local_patch, pad=None):
        """
        Perform column-wise data exchange with adjacent processes
        
        Args:
            local_patch (torch.Tensor): Local partition tensor
            pad (bool): Whether to add zero-padding for edge processes
            
        Returns:
            torch.Tensor: Tensor with exchanged column data
        """
        send_ops = []
        recv_ops = []
        left_recv = None
        right_recv = None
        
        if self.w_split > 1:
            # Send/receive left column
            if self.col_rank > 0:
                prev_rank = self.col_to_global_rank[self.col_rank - 1]
                left_col = local_patch[..., :, :1].contiguous()
                left_recv = torch.empty_like(left_col)
                send_ops.append(dist.P2POp(dist.isend, left_col, prev_rank, group=self.row_group))
                recv_ops.append(dist.P2POp(dist.irecv, left_recv, prev_rank, group=self.row_group))
                
            # Send/receive right column
            if self.col_rank < self.w_split - 1:
                next_rank = self.col_to_global_rank[self.col_rank + 1]
                right_col = local_patch[..., :, -1:].contiguous()
                right_recv = torch.empty_like(right_col)
                send_ops.append(dist.P2POp(dist.isend, right_col, next_rank, group=self.row_group))
                recv_ops.append(dist.P2POp(dist.irecv, right_recv, next_rank, group=self.row_group))
                
            # Execute communication
            reqs = dist.batch_isend_irecv(send_ops + recv_ops)
            for req in reqs:
                req.wait()
                
        # Handle padding for edge cases
        if pad:
            left_pad = torch.zeros_like(local_patch[..., :, :1]) if self.col_rank == 0 else left_recv
            right_pad = torch.zeros_like(local_patch[..., :, -1:]) if self.col_rank == self.w_split - 1 else right_recv
            return torch.cat([left_pad, local_patch, right_pad], dim=-1).contiguous()
        else:
            if self.w_split > 1:
                if self.col_rank == 0:
                    return torch.cat([local_patch, right_recv], dim=-1).contiguous()
                elif self.col_rank == self.w_split - 1:
                    return torch.cat([left_recv, local_patch], dim=-1).contiguous()
                else:
                    return torch.cat([left_recv, local_patch, right_recv], dim=-1).contiguous()
            else:
                return local_patch

    def exchange_rows(self, local_patch, pad=None):
        """
        Perform row-wise data exchange with adjacent processes
        
        Args:
            local_patch (torch.Tensor): Local partition tensor
            pad (bool): Whether to add zero-padding for edge processes
            
        Returns:
            torch.Tensor: Tensor with exchanged row data
        """
        send_ops = []
        recv_ops = []
        top_recv = None
        bottom_recv = None
        
        if self.h_split > 1:
            # Send/receive top row
            if self.row_rank > 0:
                prev_rank = self.row_to_global_rank[self.row_rank - 1]
                top_row = local_patch[..., :1, :].contiguous()
                top_recv = torch.empty_like(top_row)
                send_ops.append(dist.P2POp(dist.isend, top_row, prev_rank, group=self.col_group))
                recv_ops.append(dist.P2POp(dist.irecv, top_recv, prev_rank, group=self.col_group))
                
            # Send/receive bottom row
            if self.row_rank < self.h_split - 1:
                next_rank = self.row_to_global_rank[self.row_rank + 1]
                bottom_row = local_patch[..., -1:, :].contiguous()
                bottom_recv = torch.empty_like(bottom_row)
                send_ops.append(dist.P2POp(dist.isend, bottom_row, next_rank, group=self.col_group))
                recv_ops.append(dist.P2POp(dist.irecv, bottom_recv, next_rank, group=self.col_group))
                
            # Execute communication
            reqs = dist.batch_isend_irecv(send_ops + recv_ops)
            for req in reqs:
                req.wait()
                
        # Handle padding for edge cases
        if pad:
            top_pad = torch.zeros_like(local_patch[..., :1, :]) if self.row_rank == 0 else top_recv
            bottom_pad = torch.zeros_like(local_patch[..., -1:, :]) if self.row_rank == self.h_split - 1 else bottom_recv
            return torch.cat([top_pad, local_patch, bottom_pad], dim=-2).contiguous()
        else:
            if self.h_split > 1:
                if self.row_rank == 0:
                    return torch.cat([local_patch, bottom_recv], dim=-2).contiguous()
                elif self.row_rank == self.h_split - 1:
                    return torch.cat([top_recv, local_patch], dim=-2).contiguous()
                else:
                    return torch.cat([top_recv, local_patch, bottom_recv], dim=-2).contiguous()
            else:
                return local_patch

    def wraps_f_conv3d(self, f_conv3d=F.conv3d):
        """
        Decorator to handle distributed 3D convolution with padding
        
        Args:
            f_conv3d: Original convolution function
            
        Returns:
            Wrapped convolution function with distributed padding handling
        """
        self.ori_conv3d = f_conv3d
        
        def wrapped_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            # Process padding parameters
            if isinstance(padding, int):
                padding = (padding, padding, padding)
            else:
                padding = tuple(padding)
                if len(padding) != 3:
                    raise ValueError("padding must be an int or a 3-element tuple")
                    
            # Validate parameters
            if padding[-1] not in {0, 1} or padding[-2] not in {0, 1}:
                raise NotImplementedError("Only support padding[1]/padding[2] as 0 or 1")
            if not (all(s == 1 for s in (stride if isinstance(stride, tuple) else (stride,))) and
                    all(d == 1 for d in (dilation if isinstance(dilation, tuple) else (dilation,)))):
                raise NotImplementedError("Only support stride=1 and dilation=1")

            # Validate kernel size and padding relationship [[3]][[6]]
            kernel_size = weight.shape[2:5]  # Get kernel dimensions (depth, height, width)
            if padding[1] * 2 + 1 != kernel_size[1] or padding[2] * 2 + 1 != kernel_size[2]:
                raise ValueError(
                    f"3D Convolution requires: "
                    f"padding[1]*2+1 == kernel_size[1] and padding[2]*2+1 == kernel_size[2]. "
                    f"Got padding={padding}, kernel_size={kernel_size}"
                )
                
            # Handle row and column exchanges for padding
            if padding[-2] == 1:
                input = self.exchange_rows(input, pad=True)
            if padding[-1] == 1:
                input = self.exchange_columns(input, pad=True)
                
            # Call original convolution with adjusted padding
            return self.ori_conv3d(input, weight, bias, stride=1, padding=(padding[0],0,0), 
                                   dilation=1, groups=groups)
        return wrapped_conv3d

    def wraps_f_conv2d(self, f_conv2d=F.conv2d):
        """
        Decorator to handle distributed 2D convolution with padding
        
        Args:
            f_conv2d: Original 2D convolution function
            
        Returns:
            Wrapped 2D convolution function with distributed padding handling
        """
        self.ori_conv2d = f_conv2d
        
        def wrapped_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            # Process padding parameters
            if isinstance(padding, int):
                padding = (padding, padding)
            else:
                padding = tuple(padding)
                if len(padding) != 2:
                    raise ValueError("padding must be an int or a 2-element tuple")
                    
            # Validate parameters
            if padding[-1] not in {0, 1} or padding[-2] not in {0, 1}:
                raise NotImplementedError("Only support padding values as 0 or 1")
            if not (all(s == 1 for s in (stride if isinstance(stride, tuple) else (stride,))) and
                    all(d == 1 for d in (dilation if isinstance(dilation, tuple) else (dilation,)))):
                raise NotImplementedError("Only support stride=1 and dilation=1")

            # Validate kernel size and padding relationship [[8]]
            kernel_size = weight.shape[2:4]  # Get kernel dimensions (height, width)
            if padding[0] * 2 + 1 != kernel_size[0] or padding[1] * 2 + 1 != kernel_size[1]:
                raise ValueError(
                    f"2D Convolution requires: "
                    f"padding[0]*2+1 == kernel_size[0] and padding[1]*2+1 == kernel_size[1]. "
                    f"Got padding={padding}, kernel_size={kernel_size}"
                )
                
            # Handle row and column exchanges for padding
            if padding[-2] == 1:
                input = self.exchange_rows(input, pad=True)
            if padding[-1] == 1:
                input = self.exchange_columns(input, pad=True)
                
            # Call original convolution with adjusted padding
            return self.ori_conv2d(
                input, weight, bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=groups
            )
        return wrapped_conv2d

    def wraps_f_interpolate(self, f_interpolate=F.interpolate):
        """
        Decorator to handle distributed interpolation operations
        
        Args:
            f_interpolate: Original interpolation function
            
        Returns:
            Wrapped interpolation function with distributed handling
        """
        self.ori_interpolate = f_interpolate
        
        def wrapped_interpolate(input, size=None, scale_factor=None, mode='nearest', 
                                align_corners=None, recompute_scale_factor=None, antialias=False):
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                raise TypeError("Input must be a PyTorch Tensor.")
            if scale_factor is None:
                raise ValueError("scale_factor must be provided")
                
            spatial_dims = input.dim() - 2
            if isinstance(scale_factor, int):
                scale_factor = (scale_factor,) * spatial_dims
            if not isinstance(scale_factor, tuple) or len(scale_factor) != spatial_dims:
                raise ValueError(f"scale_factor must be an int or a tuple of length {spatial_dims}")
            if any(sf > 2 for sf in scale_factor):
                raise ValueError("Scale factors must not exceed 2")
                
            # Handle supported modes without data exchange
            if mode in {"nearest", 'area', 'nearest-exact'}: #
                return self.ori_interpolate(
                    input=input,
                    size=None,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    recompute_scale_factor=None,
                    antialias=False
                )
            else:
                # Handle modes requiring data exchange
                use_exchange_rows = scale_factor[-2] == 2
                use_exchange_columns = scale_factor[-1] == 2
                
                # Perform data exchange
                if use_exchange_columns:
                    input = self.exchange_columns(input, pad=False)
                if use_exchange_rows:
                    input = self.exchange_rows(input, pad=False)
                    
                # Perform interpolation
                output = self.ori_interpolate(
                    input=input,
                    size=None,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    recompute_scale_factor=None,
                    antialias=False
                )
                
                # Slice excess data
                if use_exchange_columns and self.w_split > 1:
                    if self.col_rank == 0:
                        output = output[..., :-2]
                    elif self.col_rank < self.w_split - 1:
                        output = output[..., 2:-2]
                    else:
                        output = output[..., 2:]
                        
                if use_exchange_rows:
                    if self.row_rank == 0:
                        output = output[..., :-2, :]
                    elif self.row_rank < self.h_split - 1:
                        output = output[..., 2:-2, :]
                    else:
                        output = output[..., 2:, :]
                return output
        return wrapped_interpolate

    def wraps_fa(self, fa, layout="BNSD"):
        """
        Decorator for attention functions with distributed key/value handling
        
        Args:
            fa: Original attention function
            layout (str): Tensor layout ('BNSD' or 'BSND')
            
        Returns:
            Wrapped attention function with distributed key/value handling
        """
        self.ori_fa = fa
        self.layout = layout
        
        def wrapped_fa(q, k, v, *args, **kwargs):
            # Validate layout
            if self.layout not in {"BNSD", "BSND"}:
                raise ValueError("Unsupported layout. Only 'BNSD' and 'BSND' are supported.")
                
            # Gather key shapes across processes
            local_shape = torch.tensor(k.shape, device=k.device)
            all_shapes = [torch.empty_like(local_shape) for _ in range(self.world_size)]
            dist.all_gather(all_shapes, local_shape)
            all_shapes = [tuple(shape.tolist()) for shape in all_shapes]
            
            # Prepare buffers for full keys/values
            gathered_k = [torch.empty(shape, dtype=k.dtype, device=k.device) for shape in all_shapes]
            gathered_v = [torch.empty_like(k_tensor) for k_tensor in gathered_k]
            
            # Gather full keys and values
            dist.all_gather(gathered_k, k.contiguous())
            dist.all_gather(gathered_v, v.contiguous())
            
            # Concatenate along sequence dimension
            if layout == "BNSD":
                full_k = torch.cat(gathered_k, dim=2)
                full_v = torch.cat(gathered_v, dim=2)
            else:
                full_k = torch.cat(gathered_k, dim=1)
                full_v = torch.cat(gathered_v, dim=1)
                
            # Call original attention function
            return self.ori_fa(q, full_k, full_v, *args, **kwargs)
        return wrapped_fa

    def wraps_decoder_fw(self, decoder_fw):
        def wrapped_decoder_fw(input, *args,**kwargs):
            input = self.patch(input)
            output = decoder_fw(input, *args,**kwargs)
            return self.dispatch(output)
        return wrapped_decoder_fw

VAE_PATCH_PARALLEL = None
FA_LAYOUT = None

def set_vae_patch_parallel(vae,h_split=1, w_split=1, world_size=None, fa_layout="BNSD",decoder_decode="decoder.forward"):
    global VAE_PATCH_PARALLEL
    global FA_LAYOUT
    VAE_PATCH_PARALLEL = Parallel_VAE_SP(h_split, w_split, world_size)
    FA_LAYOUT = fa_layout

    # wraps_decoder_fw
    decoder_decode_lst = decoder_decode.split(".")
    # the function
    ori_decoder_decode_func = reduce(getattr, decoder_decode_lst, vae)
    # the name of the function
    decoder_decode_func = decoder_decode_lst.pop()
    ori_vae_decoder = reduce(getattr, decoder_decode_lst, vae)

    new_decoder_decode = VAE_PATCH_PARALLEL.wraps_decoder_fw(ori_decoder_decode_func)
    setattr(ori_vae_decoder, decoder_decode_func, new_decoder_decode)
    return ori_decoder_decode_func

def get_vae_patch_parallel():
    return VAE_PATCH_PARALLEL

class VAE_patch_parallel:
    def __init__(self):
        global VAE_PATCH_PARALLEL
        self.vae_pp_cls = VAE_PATCH_PARALLEL
    def __enter__(self):
        if self.vae_pp_cls is not None:
            self._sub_F_func()
            self._sub_FA()

    def __exit__(self,t,v,trace):
        if self.vae_pp_cls is not None:
            self._revert_F_func()
            self._revert_FA()

    def _sub_F_func(self):
        F.conv3d = self.vae_pp_cls.wraps_f_conv3d(F.conv3d)
        F.conv2d = self.vae_pp_cls.wraps_f_conv2d(F.conv2d)
        F.interpolate = self.vae_pp_cls.wraps_f_interpolate(F.interpolate)
    def _sub_FA(self):
        global FA_LAYOUT
        F.scaled_dot_product_attention = self.vae_pp_cls.wraps_fa(
            F.scaled_dot_product_attention, layout=FA_LAYOUT)
    
    def _revert_F_func(self):
        """Restore original PyTorch functions after context exit"""
        if self.vae_pp_cls.ori_conv3d is not None:
            F.conv3d = self.vae_pp_cls.ori_conv3d
        if self.vae_pp_cls.ori_conv2d is not None:
            F.conv2d = self.vae_pp_cls.ori_conv2d
        if self.vae_pp_cls.ori_interpolate is not None:
            F.interpolate = self.vae_pp_cls.ori_interpolate
    def _revert_FA(self):
        """Restore original attention function after context exit"""
        if self.vae_pp_cls.ori_fa is not None:
            F.scaled_dot_product_attention = self.vae_pp_cls.ori_fa