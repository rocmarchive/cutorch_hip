/***************************************************************************
*     2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/
#pragma once

#include "../device_vector.h"

#include <cstddef>
#include <iterator>

namespace bolt
{
    namespace amp
    {
        template<typename T>
        class Ubiquitous_iterator : public std::iterator<device_vector_tag, T> {

            friend
            inline
            Ubiquitous_iterator operator+(
                Ubiquitous_iterator x, std::ptrdiff_t dx) [[cpu]][[hc]]
            {
                return x += dx;
            }

            friend
            inline
            std::ptrdiff_t operator-(
                const Ubiquitous_iterator& x,
                const Ubiquitous_iterator& y) [[cpu]][[hc]]
            {
                return x.p_ - y.p_;
            }

            T* p_;
        public:
            Ubiquitous_iterator() [[cpu]][[hc]] = default;
            explicit
            Ubiquitous_iterator(T* p) [[cpu]][[hc]] : p_{p} {}

            T& operator[](std::ptrdiff_t dx) const [[cpu]][[hc]]
            {
                return p_[dx];
            }
            T& operator[](std::ptrdiff_t dx) [[cpu]][[hc]] { return p_[dx]; }

            Ubiquitous_iterator& operator+=(std::ptrdiff_t dx) [[cpu]][[hc]]
            {
                p_ += dx;
                return *this;
            }

            // Bolt glue.
            std::ptrdiff_t m_Index = 0;
            const Ubiquitous_iterator& getContainer() const { return *this; }
            Ubiquitous_iterator& getContainer() { return *this; }

            T* data() const { return p_; }
            T* data() { return p_; }
            // Bolt glue.
        };
        template<typename T>
        inline
        Ubiquitous_iterator<T> make_ubiquitous_iterator(T* p) [[cpu]][[hc]]
        {
            return Ubiquitous_iterator<T>{p};
        }
    }
}
